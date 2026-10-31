#include "cpu/control.h"

#include "shared/parallel.h"
#include "shared/sparse.h"
#include "shared/vector.h"

#include <float.h>
#include <math.h>
#include <string.h>

static double clamp_double(double value, double lower, double upper)
{
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static double sigma_objective(double x, double a, double b, double c, double d)
{
    return a * x + b / x + c * x * x / (1.0 + d * x);
}

static double sigma_objective_diag(double x,
                                   double a,
                                   double b,
                                   const double *diag_q,
                                   const double *c,
                                   const double *d,
                                   double *tempv,
                                   size_t n)
{
    size_t i;
    double dot_value = 0.0;

    for (i = 0; i < n; ++i) {
        tempv[i] = d[i] / (1.0 + x * diag_q[i]);
        dot_value += c[i] * tempv[i];
    }

    return a * x + b / x + x * x * dot_value;
}

static double golden_search_sigma(double a, double b, double c, double d)
{
    double left = 1e-12;
    double right = 1e12;
    double inv_phi = (sqrt(5.0) - 1.0) / 2.0;
    double inv_phi_sq = (3.0 - sqrt(5.0)) / 2.0;
    double x1 = left + inv_phi_sq * (right - left);
    double x2 = left + inv_phi * (right - left);
    double f1 = sigma_objective(x1, a, b, c, d);
    double f2 = sigma_objective(x2, a, b, c, d);
    size_t iter;

    for (iter = 0; iter < 256; ++iter) {
        if (fabs(right - left) <= 1e-13 * fmax(1.0, fmax(fabs(left), fabs(right)))) {
            break;
        }
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + inv_phi * (right - left);
            f2 = sigma_objective(x2, a, b, c, d);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = left + inv_phi_sq * (right - left);
            f1 = sigma_objective(x1, a, b, c, d);
        }
    }

    return 0.5 * (left + right);
}

static double golden_search_sigma_diag(double a,
                                       double b,
                                       const double *diag_q,
                                       const double *c,
                                       const double *d,
                                       double *tempv,
                                       size_t n)
{
    double left = 1e-12;
    double right = 1e12;
    double phi = (sqrt(5.0) - 1.0) / 2.0;
    double x1 = right - phi * (right - left);
    double x2 = left + phi * (right - left);
    double f1 = sigma_objective_diag(x1, a, b, diag_q, c, d, tempv, n);
    double f2 = sigma_objective_diag(x2, a, b, diag_q, c, d, tempv, n);
    size_t iter = 0;

    while (fabs(right - left) > 1e-12 * fmax(1.0, fabs(left)) && iter < 200) {
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + phi * (right - left);
            f2 = sigma_objective_diag(x2, a, b, diag_q, c, d, tempv, n);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = right - phi * (right - left);
            f1 = sigma_objective_diag(x1, a, b, diag_q, c, d, tempv, n);
        }
        iter += 1;
    }

    return f1 < f2 ? x1 : x2;
}

static void update_diag_q_factors(HPRQPWorkspace *workspace)
{
    size_t i;
    for (i = 0; i < workspace->n; ++i) {
        double temp = 1.0 + workspace->sigma * workspace->diag_Q[i];
        double inv = 1.0 / temp;
        workspace->fact2[i] = inv;
        workspace->fact[i] = workspace->sigma * inv;
        workspace->fact1[i] = workspace->sigma * workspace->diag_Q[i] * inv;
        workspace->fact_M[i] = workspace->sigma * workspace->sigma * inv;
    }
}

void restart_init(HPRQPRestart *restart_info)
{
    if (restart_info == NULL) {
        return;
    }
    memset(restart_info, 0, sizeof(*restart_info));
    restart_info->first_restart = 1;
    restart_info->save_gap = INFINITY;
    restart_info->current_gap = INFINITY;
    restart_info->last_gap = INFINITY;
    restart_info->best_gap = INFINITY;
    restart_info->best_kkt = INFINITY;
    restart_info->best_sigma = 1.0;
    restart_info->sigma_correction_hold = -1;
    restart_info->weighted_norm = INFINITY;
}

double compute_m_norm(HPRQPWorkspace *workspace)
{
    HPRQPMNormTerms terms;

    if (compute_m_norm_terms(workspace, &terms) != HPRQP_STATUS_OK) {
        return INFINITY;
    }

    return terms.total;
}

HPRQPStatus compute_m_norm_terms(HPRQPWorkspace *workspace, HPRQPMNormTerms *terms)
{
    double m1 = 0.0;
    double m2;
    double m3 = 0.0;
    double dx_sq;
    double dqw_dx;
    double dy_sq = 0.0;
    double dqw_atdy = 0.0;
    double atdy_dx = 0.0;
    double dqw_sq_or_dw_dqw = 0.0;
    double atdy_qatdy = 0.0;

    if (workspace == NULL || terms == NULL || workspace->Q == NULL || workspace->AT == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    memset(terms, 0, sizeof(*terms));

    dx_sq = vec_dot(workspace->dx, workspace->dx, workspace->n);
    m2 = (1.0 / workspace->sigma) * dx_sq;
    if (matrix_csr_matvec(workspace->Q, workspace->dw, workspace->dQw) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    dqw_dx = vec_dot(workspace->dQw, workspace->dx, workspace->n);
    m2 -= 2.0 * dqw_dx;

    if (workspace->m > 0) {
        dy_sq = vec_dot(workspace->dy, workspace->dy, workspace->m);
        m1 = workspace->sigma * workspace->lambda_max_A * dy_sq;
        if (matrix_csr_matvec(workspace->AT, workspace->dy, workspace->ATdy) != HPRQP_STATUS_OK ||
            matrix_csr_matvec(workspace->Q, workspace->ATdy, workspace->QATdy) != HPRQP_STATUS_OK) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        dqw_atdy = vec_dot(workspace->dQw, workspace->ATdy, workspace->n);
        m1 -= 2.0 * workspace->sigma * dqw_atdy;
        atdy_dx = vec_dot(workspace->ATdy, workspace->dx, workspace->n);
        m2 += 2.0 * atdy_dx;

        if (workspace->Q_is_diag) {
            size_t i;
            for (i = 0; i < workspace->n; ++i) {
                workspace->tempv[i] = workspace->ATdy[i] * workspace->fact_M[i];
            }
            atdy_qatdy = vec_dot(workspace->tempv, workspace->QATdy, workspace->n);
            m3 = atdy_qatdy;
            dqw_sq_or_dw_dqw = vec_dot(workspace->dQw, workspace->dQw, workspace->n);
            m1 += workspace->sigma * dqw_sq_or_dw_dqw;
        } else {
            atdy_qatdy = vec_dot(workspace->ATdy, workspace->QATdy, workspace->n);
            m3 = (workspace->sigma * workspace->sigma) / (1.0 + workspace->sigma * workspace->lambda_max_Q) *
                 atdy_qatdy;
            dqw_sq_or_dw_dqw = vec_dot(workspace->dw, workspace->dQw, workspace->n);
            m1 += workspace->sigma * workspace->lambda_max_Q * dqw_sq_or_dw_dqw;
        }
    } else if (!workspace->Q_is_diag) {
        dqw_sq_or_dw_dqw = vec_dot(workspace->dw, workspace->dQw, workspace->n);
        m1 = workspace->sigma * workspace->lambda_max_Q * dqw_sq_or_dw_dqw;
    }

    m2 += m1 > 0.0 ? m1 : 0.0;

    terms->dx_sq = dx_sq;
    terms->dqw_dx = dqw_dx;
    terms->dy_sq = dy_sq;
    terms->dqw_atdy = dqw_atdy;
    terms->atdy_dx = atdy_dx;
    terms->dqw_sq_or_dw_dqw = dqw_sq_or_dw_dqw;
    terms->atdy_qatdy = atdy_qatdy;
    terms->m1 = m1;
    terms->m2 = m2;
    terms->m3 = m3;
    terms->total = sqrt((m2 > 0.0 ? m2 : 0.0) + (m3 > 0.0 ? m3 : 0.0));
    return HPRQP_STATUS_OK;
}

void restart_check(HPRQPRestart *restart_info, size_t iter, size_t check_iter, double sigma)
{
    double fact;

    if (restart_info == NULL || check_iter == 0) {
        return;
    }

    restart_info->restart_flag = 0;
    if (restart_info->first_restart) {
        if (iter == check_iter) {
            restart_info->first_restart = 0;
            restart_info->restart_flag = 1;
            restart_info->weighted_norm = restart_info->current_gap;
            restart_info->best_gap = restart_info->current_gap;
            restart_info->best_sigma = sigma;
        }
        return;
    }

    if (iter % check_iter != 0) {
        return;
    }

    if (restart_info->current_gap < 0.0) {
        restart_info->current_gap = 1e-6;
    }

    if (restart_info->current_gap <= 0.2 * restart_info->last_gap) {
        restart_info->sufficient += 1;
        restart_info->restart_flag = 1;
    }

    if (restart_info->current_gap <= 0.8 * restart_info->last_gap &&
        restart_info->current_gap > restart_info->save_gap) {
        restart_info->necessary += 1;
        restart_info->restart_flag = 2;
    }

    fact = restart_info->weighted_norm > 0.0 && restart_info->current_gap / restart_info->weighted_norm > 1e-1 ? 0.5 : 0.2;
    if ((double)restart_info->inner >= fact * (double)iter) {
        restart_info->long_restart += 1;
        restart_info->restart_flag = 3;
    }

    if (restart_info->best_gap > restart_info->current_gap) {
        restart_info->best_gap = restart_info->current_gap;
        restart_info->best_sigma = sigma;
    }

    restart_info->save_gap = restart_info->current_gap;
}

static int should_refresh_best_sigma_socp(double current_gap,
                                          double best_gap,
                                          double current_kkt,
                                          double best_kkt,
                                          size_t best_iter)
{
    const int improved_kkt = current_kkt < best_kkt;
    const int compatible_gap = current_gap < best_gap &&
                               current_kkt <= 1.01 * best_kkt;
    const int periodic_refresh = best_iter >= 10U &&
                                 current_kkt <= 1.2 * best_kkt;
    return improved_kkt || compatible_gap || periodic_refresh;
}

void restart_check_socp(HPRQPRestart *restart_info,
                        size_t iter,
                        size_t check_iter,
                        double sigma,
                        double current_kkt)
{
    const double long_fraction = 0.15;
    const size_t fractional_window =
        (size_t)ceil(long_fraction * (double)iter);
    const size_t long_window = fractional_window > 500U
                                   ? fractional_window
                                   : 500U;

    if (restart_info == NULL || check_iter == 0U) {
        return;
    }

    restart_info->restart_flag = 0;
    if (restart_info->first_restart) {
        if (iter == check_iter) {
            restart_info->first_restart = 0;
            restart_info->restart_flag = 1;
            restart_info->weighted_norm = restart_info->current_gap;
            restart_info->best_gap = restart_info->current_gap;
            restart_info->best_kkt = current_kkt;
            restart_info->best_sigma = sigma;
        }
        return;
    }

    if (iter % check_iter != 0U) {
        return;
    }
    if (restart_info->current_gap < 0.0) {
        restart_info->current_gap = 1e-6;
    }

    if (restart_info->current_gap <= 0.36 * restart_info->last_gap) {
        restart_info->sufficient += 1U;
        restart_info->restart_flag = 1;
    }
    if (restart_info->current_gap <= 0.8 * restart_info->last_gap &&
        restart_info->current_gap > restart_info->save_gap) {
        restart_info->necessary += 1U;
        restart_info->restart_flag = 2;
    }
    if (restart_info->inner >= long_window &&
        restart_info->current_gap > 0.9 * restart_info->save_gap) {
        restart_info->long_restart += 1U;
        restart_info->restart_flag = 3;
    }

    if (should_refresh_best_sigma_socp(restart_info->current_gap,
                                       restart_info->best_gap,
                                       current_kkt,
                                       restart_info->best_kkt,
                                       restart_info->best_iter)) {
        restart_info->best_gap = restart_info->current_gap;
        restart_info->best_kkt = current_kkt;
        restart_info->best_sigma = sigma;
        restart_info->best_iter = 0U;
    }
    restart_info->save_gap = restart_info->current_gap;
}

void restart_post_iteration(HPRQPRestart *restart_info,
                            size_t iter,
                            size_t check_iter,
                            double gap)
{
    int need_gap;

    if (restart_info == NULL) {
        return;
    }

    need_gap = restart_info->restart_flag > 0 ||
               (check_iter > 0 && (iter + 1) % check_iter == 0);
    if (need_gap) {
        if (restart_info->restart_flag > 0) {
            restart_info->last_gap = gap;
        }
        if (check_iter > 0 && (iter + 1) % check_iter == 0) {
            restart_info->current_gap = gap;
        }
    }
}

HPRQPStatus restart_post_metric(HPRQPRestart *restart_info,
                                HPRQPWorkspace *workspace,
                                size_t iter,
                                size_t check_iter,
                                double gap)
{
    if (restart_info == NULL || workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (restart_info->metric_reseed_pending) {
        if (!(isfinite(gap) && gap >= 0.0)) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        restart_info->current_gap = gap;
        restart_info->last_gap = gap;
        restart_info->weighted_norm = gap;
        restart_info->best_gap = gap;
        restart_info->best_sigma = workspace->sigma;
        restart_info->save_gap = INFINITY;
        restart_info->metric_reseed_pending = 0;
        workspace->metric_reseeds += 1;
        return HPRQP_STATUS_OK;
    }

    restart_post_iteration(restart_info, iter, check_iter, gap);
    return HPRQP_STATUS_OK;
}

HPRQPStatus restart_post_metric_socp(HPRQPRestart *restart_info,
                                     HPRQPWorkspace *workspace,
                                     size_t iter,
                                     size_t check_iter,
                                     double gap)
{
    if (restart_info == NULL || workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (restart_info->metric_reseed_pending) {
        return restart_post_metric(restart_info,
                                   workspace,
                                   iter,
                                   check_iter,
                                   gap);
    }

    if (restart_info->restart_flag > 0) {
        restart_info->last_gap = fmin(gap, restart_info->last_gap);
    }
    if (check_iter > 0U && (iter + 1U) % check_iter == 0U) {
        restart_info->current_gap = gap;
    }
    return HPRQP_STATUS_OK;
}

static HPRQPStatus refresh_parallel_workspace_coefficients(HPRQPWorkspace *workspace)
{
    return parallel_compute_coefficients(
        workspace->m,
        workspace->sigma,
        workspace->lambda_max_A,
        workspace->lambda_max_Q,
        workspace->lambda_max_AQA,
        workspace->lambda_max_AQ2A,
        workspace->zeta_AQ,
        workspace->tau,
        workspace->parallel_majorizer_mode,
        &workspace->d_y,
        &workspace->d_w,
        &workspace->parallel_y_rho,
        &workspace->parallel_y_inv,
        &workspace->parallel_w_gamma,
        &workspace->parallel_w_inv);
}

static int scalar_changed(double old_value, double new_value)
{
    double tolerance = 8.0 * DBL_EPSILON * fmax(1.0, fabs(old_value));
    return fabs(old_value - new_value) > tolerance;
}

static HPRQPStatus update_tau_parallel(const HPRQPParams *params,
                                       HPRQPRestart *restart_info,
                                       HPRQPWorkspace *workspace,
                                       int *metric_changed)
{
    double tau_old;
    double tau_new;

    *metric_changed = 0;
    if (restart_info->restart_flag < 1 || workspace->m == 0) {
        return HPRQP_STATUS_OK;
    }

    tau_old = workspace->tau;
    if (params->tau_mode != HPRQP_TAU_MODE_ADAPTIVE_RESTART) {
        tau_new = parallel_select_tau(params,
                                      workspace->lambda_max_A,
                                      workspace->lambda_max_Q);
    } else if (workspace->parallel_majorizer_mode ==
                   HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER &&
               workspace->zeta_AQ == 0.0) {
        tau_new = parallel_select_tau(params,
                                      workspace->lambda_max_A,
                                      workspace->lambda_max_Q);
    } else {
        double dy_sq = vec_dot(workspace->dy, workspace->dy, workspace->m);
        double dw_dqw = vec_dot(workspace->dw,
                                workspace->dQw,
                                workspace->n);
        if (workspace->parallel_majorizer_mode ==
            HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT) {
            tau_new = parallel_select_two_moment_tau(
                params,
                workspace->lambda_max_A,
                workspace->lambda_max_Q,
                workspace->lambda_max_AQA,
                workspace->lambda_max_AQ2A,
                dy_sq,
                dw_dqw);
        } else {
            double y_energy = workspace->lambda_max_A * dy_sq;
            double w_energy = workspace->lambda_max_Q * dw_dqw;
            double energy_floor = params->parallel_adaptation_energy_floor;
            if (!(isfinite(y_energy) && y_energy > energy_floor &&
                  isfinite(w_energy) && w_energy > energy_floor)) {
                tau_new = parallel_select_tau(params,
                                              workspace->lambda_max_A,
                                              workspace->lambda_max_Q);
            } else {
                double tau_raw = sqrt(w_energy / y_energy);
                tau_new = isfinite(tau_raw) && tau_raw > 0.0
                              ? clamp_double(tau_raw,
                                             params->tau_min,
                                             params->tau_max)
                              : parallel_select_tau(params,
                                                    workspace->lambda_max_A,
                                                    workspace->lambda_max_Q);
            }
        }
    }

    if (!scalar_changed(tau_old, tau_new)) {
        return HPRQP_STATUS_OK;
    }
    workspace->tau = tau_new;
    if (refresh_parallel_workspace_coefficients(workspace) !=
        HPRQP_STATUS_OK) {
        workspace->tau = tau_old;
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *metric_changed = !(workspace->parallel_majorizer_mode ==
                            HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER &&
                        workspace->zeta_AQ == 0.0);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus parallel_metric_sq_checked(const HPRQPWorkspace *workspace,
                                              double term_y,
                                              double term_w,
                                              double term_cross,
                                              double term_x,
                                              double *metric_sq)
{
    double value;
    double scale;
    double roundoff_floor;

    if (workspace == NULL || metric_sq == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    value = term_y + term_w + term_cross + term_x;
    scale = fabs(term_y) + fabs(term_w) + fabs(term_cross) + fabs(term_x);
    roundoff_floor = workspace->metric_negative_tolerance * fmax(1.0, scale);
    if (!isfinite(value) || value < -roundoff_floor) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *metric_sq = value > 0.0 ? value : 0.0;
    return HPRQP_STATUS_OK;
}

double compute_m_norm_parallel(HPRQPWorkspace *workspace)
{
    double term_y;
    double term_w;
    double term_cross;
    double term_x;
    double metric_sq;

    if (workspace == NULL || !(workspace->sigma > 0.0)) {
        return INFINITY;
    }
    term_y = workspace->m > 0
                 ? workspace->sigma * workspace->d_y *
                       workspace->lambda_max_A *
                       vec_dot(workspace->dy, workspace->dy, workspace->m)
                 : 0.0;
    term_w = workspace->sigma * workspace->d_w *
             workspace->lambda_max_Q *
             vec_dot(workspace->dw, workspace->dQw, workspace->n);
    term_cross = 2.0 *
                 (vec_dot(workspace->ATdy, workspace->dx, workspace->n) -
                  vec_dot(workspace->dQw, workspace->dx, workspace->n));
    term_x = vec_dot(workspace->dx, workspace->dx, workspace->n) /
             workspace->sigma;
    if (parallel_metric_sq_checked(workspace,
                                   term_y,
                                   term_w,
                                   term_cross,
                                   term_x,
                                   &metric_sq) != HPRQP_STATUS_OK) {
        return INFINITY;
    }
    return sqrt(metric_sq);
}

static HPRQPStatus update_sigma_parallel(const HPRQPParams *params,
                                         HPRQPRestart *restart_info,
                                         HPRQPWorkspace *workspace,
                                         const HPRQPResiduals *residuals,
                                         int *sigma_changed)
{
    size_t i;
    double sigma_old;
    double a_M;
    double b_M;
    double c_M;
    double metric_sq;
    double sigma_estimation;
    double fact;
    double sigma_cand;
    double temp_1;
    double ratio_infeas_org;
    double kappa;
    double sigma_new;

    *sigma_changed = 0;
    if (params->sigma_fixed || restart_info->restart_flag < 1) {
        return HPRQP_STATUS_OK;
    }

    sigma_old = workspace->sigma;
    for (i = 0; i < workspace->n; ++i) {
        workspace->dx[i] = workspace->x_bar[i] - workspace->last_x[i];
        workspace->dw[i] = workspace->w_bar[i] - workspace->last_w[i];
        workspace->dQw[i] = workspace->Qw_bar[i] - workspace->last_Qw[i];
    }
    a_M = workspace->d_w * workspace->lambda_max_Q *
          vec_dot(workspace->dw, workspace->dQw, workspace->n);
    c_M = -vec_dot(workspace->dQw, workspace->dx, workspace->n);
    if (workspace->m > 0) {
        for (i = 0; i < workspace->m; ++i) {
            workspace->dy[i] = workspace->y_bar[i] - workspace->last_y[i];
        }
        for (i = 0; i < workspace->n; ++i) {
            workspace->ATdy[i] = workspace->ATy_bar[i] -
                                  workspace->last_ATy[i];
        }
        a_M += workspace->d_y * workspace->lambda_max_A *
               vec_dot(workspace->dy, workspace->dy, workspace->m);
        c_M += vec_dot(workspace->ATdy, workspace->dx, workspace->n);
    } else {
        vec_fill(workspace->ATdy, workspace->n, 0.0);
    }
    b_M = vec_dot(workspace->dx, workspace->dx, workspace->n);

    if (parallel_metric_sq_checked(workspace,
                                   sigma_old * a_M,
                                   0.0,
                                   2.0 * c_M,
                                   b_M / sigma_old,
                                   &metric_sq) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    (void)metric_sq;

    if (!(isfinite(a_M) &&
          a_M > params->parallel_adaptation_energy_floor &&
          isfinite(b_M) &&
          b_M > params->parallel_adaptation_energy_floor)) {
        return HPRQP_STATUS_OK;
    }

    sigma_estimation = sqrt(fmax(b_M, 1e-12) / fmax(a_M, 1e-12));
    fact = (restart_info->best_gap > 0.0 &&
            isfinite(restart_info->best_gap))
               ? exp(-0.05 *
                     (restart_info->current_gap / restart_info->best_gap))
               : 1.0;
    sigma_cand = exp(
        fact * log(sigma_estimation) +
        (1.0 - fact) *
            log(restart_info->best_sigma > 0.0
                    ? restart_info->best_sigma
                    : sigma_old));

    temp_1 = fmax(fmin(residuals->err_Rd_org_bar,
                       residuals->err_Rp_org_bar),
                  fmin(residuals->rel_gap_bar,
                       restart_info->current_gap));
    if (residuals->err_Rp_org_bar > 0.0) {
        ratio_infeas_org = residuals->err_Rd_org_bar /
                           residuals->err_Rp_org_bar;
    } else if (residuals->err_Rd_org_bar > 0.0) {
        ratio_infeas_org = INFINITY;
    } else {
        ratio_infeas_org = 1.0;
    }
    if (temp_1 > 9e-10) {
        kappa = 1.0;
    } else if (temp_1 > 5e-10) {
        kappa = clamp_double(sqrt(ratio_infeas_org), 1e-2, 100.0);
    } else {
        kappa = clamp_double(ratio_infeas_org, 1e-2, 100.0);
    }
    sigma_new = kappa * sigma_cand;
    if (!(isfinite(sigma_new) && sigma_new > 0.0)) {
        return HPRQP_STATUS_OK;
    }
    if (fabs(sigma_old - sigma_new) > 1e-15) {
        workspace->sigma = sigma_new;
        if (refresh_parallel_workspace_coefficients(workspace) !=
            HPRQP_STATUS_OK) {
            workspace->sigma = sigma_old;
            (void)refresh_parallel_workspace_coefficients(workspace);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        *sigma_changed = 1;
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus parallel_update_metric(const HPRQPParams *params,
                                   HPRQPRestart *restart_info,
                                   HPRQPWorkspace *workspace,
                                   const HPRQPResiduals *residuals)
{
    HPRQPStatus status;
    int tau_changed;
    int sigma_changed;

    if (params == NULL || restart_info == NULL || workspace == NULL ||
        residuals == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    status = update_tau_parallel(params,
                                 restart_info,
                                 workspace,
                                 &tau_changed);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = update_sigma_parallel(params,
                                   restart_info,
                                   workspace,
                                   residuals,
                                   &sigma_changed);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (tau_changed || sigma_changed) {
        restart_info->metric_reseed_pending = 1;
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus sigma_update(const HPRQPParams *params,
                         HPRQPRestart *restart_info,
                         HPRQPWorkspace *workspace,
                         const HPRQPResiduals *residuals)
{
    size_t i;
    double sigma_old;
    double a = 0.0;
    double b;
    double sigma_estimation;
    double sigma_cand;
    double fact;
    double temp_1;
    double kappa = 1.0;
    double ratio_infeas_org;
    double c_term = 0.0;
    double d_term = 0.0;

    if (params == NULL || restart_info == NULL || workspace == NULL || residuals == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (restart_info->restart_flag < 1) {
        return HPRQP_STATUS_OK;
    }

    sigma_old = workspace->sigma;
    for (i = 0; i < workspace->n; ++i) {
        workspace->dx[i] = workspace->x_bar[i] - workspace->last_x[i];
        workspace->dw[i] = workspace->w_bar[i] - workspace->last_w[i];
    }
    b = vec_dot(workspace->dx, workspace->dx, workspace->n);

    if (matrix_csr_matvec(workspace->Q, workspace->dw, workspace->dQw) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (workspace->m > 0) {
        for (i = 0; i < workspace->m; ++i) {
            workspace->dy[i] = workspace->y_bar[i] - workspace->last_y[i];
        }
        for (i = 0; i < workspace->n; ++i) {
            workspace->ATdy[i] = workspace->ATy_bar[i] - workspace->last_ATy[i];
        }
        a = workspace->lambda_max_A * vec_dot(workspace->dy, workspace->dy, workspace->m);
        a -= 2.0 * vec_dot(workspace->dQw, workspace->ATdy, workspace->n);
        if (matrix_csr_matvec(workspace->Q, workspace->ATdy, workspace->QATdy) != HPRQP_STATUS_OK) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
    }

    if (workspace->Q_is_diag) {
        a += vec_dot(workspace->dQw, workspace->dQw, workspace->n);
    } else {
        a += workspace->lambda_max_Q * vec_dot(workspace->dw, workspace->dQw, workspace->n);
        if (workspace->m > 0) {
            c_term = vec_dot(workspace->ATdy, workspace->QATdy, workspace->n);
            d_term = workspace->lambda_max_Q;
        }
    }

    a = a > 1e-12 ? a : 1e-12;
    b = b > 1e-12 ? b : 1e-12;
    if (workspace->Q_is_diag && workspace->m > 0) {
        sigma_estimation = golden_search_sigma_diag(a,
                                                    b,
                                                    workspace->diag_Q,
                                                    workspace->ATdy,
                                                    workspace->QATdy,
                                                    workspace->tempv,
                                                    workspace->n);
    } else if (!workspace->Q_is_diag && workspace->m > 0) {
        c_term = c_term > 0.0 ? c_term : 0.0;
        d_term = d_term > 0.0 ? d_term : 0.0;
        sigma_estimation = golden_search_sigma(a, b, c_term, d_term);
    } else {
        sigma_estimation = sqrt(b / a);
    }
    sigma_estimation = clamp_double(sigma_estimation, 1e-12, 1e12);

    fact = (restart_info->best_gap > 0.0 && isfinite(restart_info->best_gap))
               ? exp(-0.05 * (restart_info->current_gap / restart_info->best_gap))
               : 1.0;
    sigma_cand = exp(fact * log(sigma_estimation) + (1.0 - fact) * log(restart_info->best_sigma > 0.0 ? restart_info->best_sigma : sigma_old));

    temp_1 = fmax(fmin(residuals->err_Rd_org_bar, residuals->err_Rp_org_bar),
                  fmin(residuals->rel_gap_bar, restart_info->current_gap));
    if (temp_1 <= 9e-10 && residuals->err_Rp_org_bar > 0.0) {
        ratio_infeas_org = residuals->err_Rd_org_bar / residuals->err_Rp_org_bar;
        if (temp_1 > 5e-10) {
            kappa = clamp_double(sqrt(ratio_infeas_org), 1e-2, 100.0);
        } else {
            kappa = clamp_double(ratio_infeas_org, 1e-2, 100.0);
        }
    }

    workspace->sigma = kappa * sigma_cand;
    workspace->sigma = clamp_double(workspace->sigma, 1e-12, 1e12);

    if (workspace->Q_is_diag && fabs(workspace->sigma - sigma_old) > 1e-15) {
        update_diag_q_factors(workspace);
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus restart_apply(HPRQPRestart *restart_info, HPRQPWorkspace *workspace)
{
    size_t i;

    if (restart_info == NULL || workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (restart_info->restart_flag <= 0) {
        return HPRQP_STATUS_OK;
    }

    for (i = 0; i < workspace->n; ++i) {
        workspace->x[i] = workspace->x_bar[i];
        workspace->w[i] = workspace->w_bar[i];
        workspace->last_x[i] = workspace->x_bar[i];
        workspace->last_w[i] = workspace->w_bar[i];
    }

    if (workspace->m > 0) {
        for (i = 0; i < workspace->m; ++i) {
            workspace->y[i] = workspace->y_bar[i];
            workspace->last_y[i] = workspace->y_bar[i];
        }
        if (matrix_csr_matvec(workspace->AT, workspace->y_bar, workspace->ATy_bar) != HPRQP_STATUS_OK) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        for (i = 0; i < workspace->n; ++i) {
            workspace->last_ATy[i] = workspace->ATy_bar[i];
            workspace->ATy[i] = workspace->ATy_bar[i];
        }
    }

    if (workspace->noC) {
        for (i = 0; i < workspace->n; ++i) {
            workspace->Qw[i] = workspace->Qw_bar[i];
            workspace->last_Qw[i] = workspace->Qw_bar[i];
        }
    }

    restart_info->last_gap = restart_info->current_gap;
    restart_info->save_gap = INFINITY;
    restart_info->times += 1;
    restart_info->inner = 0;
    return HPRQP_STATUS_OK;
}

HPRQPStatus restart_apply_parallel(HPRQPRestart *restart_info,
                                   HPRQPWorkspace *workspace)
{
    size_t i;

    if (restart_info == NULL || workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (restart_info->restart_flag <= 0) {
        return HPRQP_STATUS_OK;
    }

    for (i = 0; i < workspace->n; ++i) {
        workspace->x[i] = workspace->x_bar[i];
        workspace->w[i] = workspace->w_bar[i];
        workspace->last_x[i] = workspace->x_bar[i];
        workspace->last_w[i] = workspace->w_bar[i];
        workspace->Qw[i] = workspace->Qw_bar[i];
        workspace->last_Qw[i] = workspace->Qw_bar[i];
    }
    if (workspace->m > 0) {
        for (i = 0; i < workspace->m; ++i) {
            workspace->y[i] = workspace->y_bar[i];
            workspace->last_y[i] = workspace->y_bar[i];
        }
        for (i = 0; i < workspace->n; ++i) {
            workspace->ATy[i] = workspace->ATy_bar[i];
            workspace->last_ATy[i] = workspace->ATy_bar[i];
        }
    }

    restart_info->last_gap = restart_info->metric_reseed_pending
                                 ? INFINITY
                                 : restart_info->current_gap;
    restart_info->save_gap = INFINITY;
    restart_info->times += 1;
    restart_info->inner = 0;
    return HPRQP_STATUS_OK;
}
