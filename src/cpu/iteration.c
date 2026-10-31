#include "cpu/iteration.h"

#include "cpu/eigs.h"
#include "shared/sparse.h"

#include <math.h>

static double interval_project(double value, double lower, double upper)
{
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static HPRQPStatus update_zxw1_noC(HPRQPWorkspace *workspace, double halpern_fact1, double halpern_fact2)
{
    HPRQPStatus status;
    size_t i;
    double fact2_scalar = 1.0 / (1.0 + workspace->sigma * workspace->lambda_max_Q);
    double fact1_scalar = 1.0 - fact2_scalar;

    if (workspace->m > 0) {
        status = matrix_csr_matvec(workspace->AT, workspace->y, workspace->ATy);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    } else {
        for (i = 0; i < workspace->n; ++i) {
            workspace->ATy[i] = 0.0;
        }
    }

    for (i = 0; i < workspace->n; ++i) {
        double tmp = -workspace->Qw[i] + workspace->ATy[i] - workspace->c[i];
        double z_raw = workspace->x[i] + workspace->sigma * tmp;
        double x_bar_i = interval_project(z_raw, workspace->l[i], workspace->u[i]);
        double x_hat_i = 2.0 * x_bar_i - workspace->x[i];
        double x_new = halpern_fact2 * x_hat_i + halpern_fact1 * workspace->last_x[i];
        double w_bar_i;
        double w_hat_i;
        double w_new;

        if (workspace->Q_is_diag) {
            w_bar_i = workspace->fact1[i] * workspace->w[i] + workspace->fact2[i] * x_hat_i;
        } else {
            w_bar_i = fact1_scalar * workspace->w[i] + fact2_scalar * x_hat_i;
        }
        w_hat_i = 2.0 * w_bar_i - workspace->w[i];
        w_new = halpern_fact2 * w_hat_i + halpern_fact1 * workspace->last_w[i];

        if (workspace->to_check) {
            workspace->dx[i] = x_bar_i - workspace->x[i];
            workspace->dw[i] = w_bar_i - workspace->w[i];
            workspace->z_bar[i] = (x_bar_i - z_raw) / workspace->sigma;
        }

        workspace->x_bar[i] = x_bar_i;
        workspace->x_hat[i] = x_hat_i;
        workspace->x[i] = x_new;
        workspace->w_bar[i] = w_bar_i;
        workspace->w[i] = w_new;
    }

    status = matrix_csr_matvec(workspace->Q, workspace->w_bar, workspace->Qw_bar);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    for (i = 0; i < workspace->n; ++i) {
        double qw_i = workspace->Qw[i];
        double qw_bar_i = workspace->Qw_bar[i];
        double qw_hat_i = 2.0 * qw_bar_i - qw_i;

        workspace->Qw[i] = halpern_fact2 * qw_hat_i + halpern_fact1 * workspace->last_Qw[i];
        workspace->tempv[i] = workspace->x_hat[i] + workspace->sigma * (qw_i - qw_bar_i);
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus update_zxw1(HPRQPWorkspace *workspace, double halpern_fact1, double halpern_fact2)
{
    HPRQPStatus status;
    size_t i;
    double fact2_scalar = 1.0 / (1.0 + workspace->sigma * workspace->lambda_max_Q);
    double fact1_scalar = 1.0 - fact2_scalar;

    status = matrix_csr_matvec(workspace->Q, workspace->w, workspace->Qw);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    for (i = 0; i < workspace->n; ++i) {
        double tmp = -workspace->Qw[i] + workspace->ATy[i] - workspace->c[i];
        double z_raw = workspace->x[i] + workspace->sigma * tmp;
        double x_bar_i = interval_project(z_raw, workspace->l[i], workspace->u[i]);
        double x_hat_i = 2.0 * x_bar_i - workspace->x[i];
        double x_new = halpern_fact2 * x_hat_i + halpern_fact1 * workspace->last_x[i];
        double w_bar_i;

        if (workspace->Q_is_diag) {
            w_bar_i = workspace->fact1[i] * workspace->w[i] + workspace->fact2[i] * x_hat_i;
        } else {
            w_bar_i = fact1_scalar * workspace->w[i] + fact2_scalar * x_hat_i;
        }

        if (workspace->to_check) {
            workspace->dx[i] = x_bar_i - workspace->x[i];
            workspace->z_bar[i] = (x_bar_i - z_raw) / workspace->sigma;
        }

        workspace->x_bar[i] = x_bar_i;
        workspace->x_hat[i] = x_hat_i;
        workspace->x[i] = x_new;
        workspace->w_bar[i] = w_bar_i;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus update_y(HPRQPWorkspace *workspace, double halpern_fact1, double halpern_fact2)
{
    size_t i;
    HPRQPStatus status;
    double fact1;
    double fact2;

    if (workspace->m == 0) {
        return HPRQP_STATUS_OK;
    }

    status = matrix_csr_matvec(workspace->Q, workspace->w_bar, workspace->Qw_bar);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    for (i = 0; i < workspace->n; ++i) {
        workspace->tempv[i] = workspace->x_hat[i] + workspace->sigma * (workspace->Qw[i] - workspace->Qw_bar[i]);
    }

    status = matrix_csr_matvec(workspace->A, workspace->tempv, workspace->Ax);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    fact1 = workspace->lambda_max_A * workspace->sigma;
    if (fact1 <= 0.0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    fact2 = 1.0 / fact1;

    for (i = 0; i < workspace->m; ++i) {
        double s_raw = workspace->Ax[i] - fact1 * workspace->y[i];
        double s_proj = interval_project(s_raw, workspace->AL[i], workspace->AU[i]);
        double y_bar_i = fact2 * (s_proj - s_raw);
        double y_hat_i = 2.0 * y_bar_i - workspace->y[i];
        double y_new = halpern_fact2 * y_hat_i + halpern_fact1 * workspace->last_y[i];

        if (workspace->to_check) {
            workspace->s[i] = s_proj;
            workspace->dy[i] = y_bar_i - workspace->y[i];
        }

        workspace->y_bar[i] = y_bar_i;
        workspace->y_hat[i] = y_hat_i;
        workspace->y[i] = y_new;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus update_y_noC(HPRQPWorkspace *workspace, double halpern_fact1, double halpern_fact2)
{
    size_t i;
    HPRQPStatus status;
    double fact1;
    double fact2;

    if (workspace->m == 0) {
        return HPRQP_STATUS_OK;
    }

    status = matrix_csr_matvec(workspace->A, workspace->tempv, workspace->Ax);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    fact1 = workspace->lambda_max_A * workspace->sigma;
    if (fact1 <= 0.0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    fact2 = 1.0 / fact1;

    for (i = 0; i < workspace->m; ++i) {
        double s_raw = workspace->Ax[i] - fact1 * workspace->y[i];
        double s_proj = interval_project(s_raw, workspace->AL[i], workspace->AU[i]);
        double y_bar_i = fact2 * (s_proj - s_raw);
        double y_hat_i = 2.0 * y_bar_i - workspace->y[i];
        double y_new = halpern_fact2 * y_hat_i + halpern_fact1 * workspace->last_y[i];

        if (workspace->to_check) {
            workspace->s[i] = s_proj;
            workspace->dy[i] = y_bar_i - workspace->y[i];
        }

        workspace->y_bar[i] = y_bar_i;
        workspace->y_hat[i] = y_hat_i;
        workspace->y[i] = y_new;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus update_w2(HPRQPWorkspace *workspace, double halpern_fact1, double halpern_fact2)
{
    size_t i;
    HPRQPStatus status;
    double fact_scalar = workspace->sigma / (1.0 + workspace->sigma * workspace->lambda_max_Q);

    if (workspace->m > 0) {
        status = matrix_csr_matvec(workspace->AT, workspace->y_bar, workspace->ATy_bar);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    for (i = 0; i < workspace->n; ++i) {
        double fact = workspace->Q_is_diag ? workspace->fact[i] : fact_scalar;
        double w_bar_new = workspace->w_bar[i] + fact * (workspace->ATy_bar[i] - workspace->ATy[i]);
        double w_new = halpern_fact1 * workspace->last_w[i] + halpern_fact2 * (2.0 * w_bar_new - workspace->w[i]);
        double aty_new = halpern_fact1 * workspace->last_ATy[i] + halpern_fact2 * (2.0 * workspace->ATy_bar[i] - workspace->ATy[i]);

        if (workspace->to_check) {
            workspace->dw[i] = w_bar_new - workspace->w[i];
            workspace->ATdy[i] = workspace->ATy_bar[i] - workspace->ATy[i];
        }

        workspace->w_bar[i] = w_bar_new;
        workspace->w[i] = w_new;
        workspace->ATy[i] = aty_new;
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus iteration_step_parallel(HPRQPWorkspace *workspace,
                                    size_t inner_iter)
{
    HPRQPStatus status;
    double halpern_fact1;
    double halpern_fact2;
    size_t i;

    if (workspace == NULL || workspace->Q == NULL || workspace->A == NULL ||
        workspace->AT == NULL || workspace->c == NULL || workspace->l == NULL ||
        workspace->u == NULL || workspace->x == NULL || workspace->x_bar == NULL ||
        workspace->x_hat == NULL || workspace->w == NULL ||
        workspace->w_bar == NULL || workspace->Qw == NULL ||
        workspace->Qw_bar == NULL || workspace->ATy == NULL ||
        workspace->ATy_bar == NULL || workspace->last_x == NULL ||
        workspace->last_w == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->m > 0 &&
        (workspace->y == NULL || workspace->y_bar == NULL ||
         workspace->last_y == NULL || workspace->Ax == NULL ||
         workspace->AL == NULL || workspace->AU == NULL)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    halpern_fact1 = 1.0 / ((double)inner_iter + 2.0);
    halpern_fact2 = 1.0 - halpern_fact1;

    status = matrix_csr_matvec(workspace->Q, workspace->w, workspace->Qw);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (workspace->m > 0) {
        status = matrix_csr_matvec(workspace->AT,
                                   workspace->y,
                                   workspace->ATy);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    for (i = 0; i < workspace->n; ++i) {
        double x_i = workspace->x[i];
        double w_i = workspace->w[i];
        double z_raw = x_i + workspace->sigma *
                                   (-workspace->Qw[i] + workspace->ATy[i] -
                                    workspace->c[i]);
        double x_bar_i = z_raw < workspace->l[i]
                             ? workspace->l[i]
                             : (z_raw > workspace->u[i] ? workspace->u[i]
                                                         : z_raw);
        double x_hat_i = 2.0 * x_bar_i - x_i;
        double w_bar_i = (workspace->parallel_w_gamma * w_i + x_hat_i) *
                         workspace->parallel_w_inv;

        workspace->x_bar[i] = x_bar_i;
        workspace->x_hat[i] = x_hat_i;
        workspace->z_bar[i] = (x_bar_i - z_raw) / workspace->sigma;
        workspace->w_bar[i] = w_bar_i;
        if (workspace->to_check) {
            workspace->dx[i] = x_bar_i - x_i;
            workspace->dw[i] = w_bar_i - w_i;
        }
        workspace->x[i] = halpern_fact2 * x_hat_i +
                          halpern_fact1 * workspace->last_x[i];
        workspace->w[i] = halpern_fact2 * (2.0 * w_bar_i - w_i) +
                          halpern_fact1 * workspace->last_w[i];
    }

    if (workspace->m > 0) {
        status = matrix_csr_matvec(workspace->A,
                                   workspace->x_hat,
                                   workspace->Ax);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        for (i = 0; i < workspace->m; ++i) {
            double y_i = workspace->y[i];
            double s_raw = workspace->Ax[i] -
                           workspace->parallel_y_rho * y_i;
            double s_proj = s_raw < workspace->AL[i]
                                ? workspace->AL[i]
                                : (s_raw > workspace->AU[i]
                                       ? workspace->AU[i]
                                       : s_raw);
            double y_bar_i = (s_proj - s_raw) * workspace->parallel_y_inv;

            workspace->s[i] = s_proj;
            workspace->y_bar[i] = y_bar_i;
            if (workspace->to_check) {
                workspace->dy[i] = y_bar_i - y_i;
            }
            workspace->y[i] = halpern_fact2 * (2.0 * y_bar_i - y_i) +
                              halpern_fact1 * workspace->last_y[i];
        }
    }

    if (workspace->to_check) {
        status = matrix_csr_matvec(workspace->Q,
                                   workspace->w_bar,
                                   workspace->Qw_bar);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        if (workspace->m > 0) {
            status = matrix_csr_matvec(workspace->AT,
                                       workspace->y_bar,
                                       workspace->ATy_bar);
            if (status != HPRQP_STATUS_OK) {
                return status;
            }
        }
        for (i = 0; i < workspace->n; ++i) {
            workspace->dQw[i] = workspace->Qw_bar[i] - workspace->Qw[i];
            workspace->ATdy[i] = workspace->m > 0
                                     ? workspace->ATy_bar[i] - workspace->ATy[i]
                                     : 0.0;
        }
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus iteration_step(HPRQPWorkspace *workspace, size_t inner_iter)
{
    HPRQPStatus status;
    double halpern_fact1;
    double halpern_fact2;

    if (workspace == NULL || workspace->Q == NULL || workspace->A == NULL || workspace->AT == NULL || workspace->c == NULL ||
        workspace->l == NULL || workspace->u == NULL || workspace->x == NULL || workspace->x_bar == NULL || workspace->w == NULL ||
        workspace->w_bar == NULL || workspace->Qw == NULL || workspace->ATy == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (!workspace->eigenvalues_initialized) {
        status = estimate_eigenvalues(workspace, NULL);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        workspace->eigenvalues_initialized = 1;
    }

    if (workspace->method == HPRQP_METHOD_PARALLEL) {
        return iteration_step_parallel(workspace, inner_iter);
    }

    halpern_fact1 = 1.0 / ((double)inner_iter + 2.0);
    halpern_fact2 = 1.0 - halpern_fact1;

    if (workspace->noC) {
        status = update_zxw1_noC(workspace, halpern_fact1, halpern_fact2);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        return update_y_noC(workspace, halpern_fact1, halpern_fact2);
    }

    status = update_zxw1(workspace, halpern_fact1, halpern_fact2);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = update_y(workspace, halpern_fact1, halpern_fact2);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = update_w2(workspace, halpern_fact1, halpern_fact2);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    return HPRQP_STATUS_OK;
}
