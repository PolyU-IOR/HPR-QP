#include "control.h"

#include "sparse.h"
#include "vector.h"

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
    restart_info->best_sigma = 1.0;
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