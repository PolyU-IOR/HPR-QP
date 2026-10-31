#include "cpu/residuals.h"

#include "shared/sparse.h"
#include "shared/vector.h"

#include <math.h>
#include <string.h>

static double max2(double a, double b)
{
    return a > b ? a : b;
}

static double max3(double a, double b, double c)
{
    return max2(max2(a, b), c);
}

static double col_scale_at(const HPRQPScaling *scaling, size_t index)
{
    if (scaling->col_norm == NULL) {
        return scaling->c_scale;
    }
    return scaling->col_norm[index] * scaling->c_scale;
}

static double row_scale_at(const HPRQPScaling *scaling, size_t index)
{
    if (scaling->row_norm == NULL) {
        return scaling->b_scale;
    }
    return scaling->row_norm[index] * scaling->b_scale;
}

static double project_interval(double value, double lower, double upper)
{
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static double compute_dual_objective(const HPRQPWorkspace *workspace,
                                     const HPRQPModel *model,
                                     const HPRQPScaling *scaling)
{
    double value = -0.5 * vec_dot(workspace->x_bar, workspace->Qx, workspace->n) +
                   vec_dot(workspace->z_bar, workspace->x_bar, workspace->n);
    if (workspace->m > 0) {
        value += vec_dot(workspace->y_bar, workspace->s, workspace->m);
    }
    return scaling->b_scale * scaling->c_scale * value + model->obj_constant;
}

static double compute_bound_violation(HPRQPWorkspace *workspace, const HPRQPScaling *scaling)
{
    size_t i;
    double max_violation = 0.0;
    for (i = 0; i < workspace->n; ++i) {
        double lower_violation = workspace->l[i] - workspace->x_bar[i];
        double upper_violation = workspace->x_bar[i] - workspace->u[i];
        double violation = 0.0;
        if (lower_violation > 0.0) {
            violation += lower_violation;
        }
        if (upper_violation > 0.0) {
            violation += upper_violation;
        }
        workspace->dx[i] = violation * (scaling->b_scale / (scaling->col_norm == NULL ? 1.0 : scaling->col_norm[i]));
        if (workspace->dx[i] > max_violation) {
            max_violation = workspace->dx[i];
        }
    }
    return max_violation;
}

void residuals_init(HPRQPResiduals *residuals)
{
    if (residuals == NULL) {
        return;
    }
    memset(residuals, 0, sizeof(*residuals));
}

HPRQPStatus residuals_compute(HPRQPWorkspace *workspace,
                              const HPRQPModel *model,
                              const HPRQPScaling *scaling,
                              HPRQPResiduals *residuals,
                              size_t iter)
{
    HPRQPStatus status;

    if (workspace == NULL || model == NULL || scaling == NULL || residuals == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    residuals_init(residuals);

    status = matrix_csr_matvec(workspace->Q, workspace->x_bar, workspace->Qx);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (workspace->m > 0) {
        status = matrix_csr_matvec(workspace->A, workspace->x_bar, workspace->Ax);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = matrix_csr_matvec(workspace->AT, workspace->y_bar, workspace->ATdy);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    } else {
        vec_fill(workspace->ATdy, workspace->n, 0.0);
    }

    return residuals_finalize_from_products(workspace, model, scaling, residuals, iter);
}

HPRQPStatus residuals_finalize_from_products(HPRQPWorkspace *workspace,
                                             const HPRQPModel *model,
                                             const HPRQPScaling *scaling,
                                             HPRQPResiduals *residuals,
                                             size_t iter)
{
    size_t i;
    double max_scaled_qx = 0.0;
    double max_scaled_atdy = 0.0;
    double max_scaled_ax = 0.0;

    if (workspace == NULL || model == NULL || scaling == NULL || residuals == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    residuals_init(residuals);

    residuals->primal_obj_bar = scaling->b_scale * scaling->c_scale *
                                (vec_dot(workspace->c, workspace->x_bar, workspace->n) +
                                 0.5 * vec_dot(workspace->x_bar, workspace->Qx, workspace->n)) +
                                model->obj_constant;
    residuals->dual_obj_bar = compute_dual_objective(workspace, model, scaling);
    residuals->rel_gap_bar = fabs(residuals->primal_obj_bar - residuals->dual_obj_bar) /
                             (1.0 + max2(fabs(residuals->primal_obj_bar), fabs(residuals->dual_obj_bar)));

    for (i = 0; i < workspace->n; ++i) {
        double scale_fact = col_scale_at(scaling, i);
        double scaled_qx = workspace->Qx[i] * scale_fact;
        double scaled_atdy = workspace->ATdy[i] * scale_fact;
        workspace->Rd[i] = (workspace->Qx[i] + workspace->c[i] - workspace->ATdy[i] - workspace->z_bar[i]) * scale_fact;
        if (fabs(scaled_qx) > max_scaled_qx) {
            max_scaled_qx = fabs(scaled_qx);
        }
        if (fabs(scaled_atdy) > max_scaled_atdy) {
            max_scaled_atdy = fabs(scaled_atdy);
        }
    }

    residuals->err_Rd_org_bar = vec_norm_inf(workspace->Rd, workspace->n) /
                                (1.0 + max3(scaling->norm_c_org, max_scaled_atdy, max_scaled_qx));

    if (workspace->m > 0) {
        for (i = 0; i < workspace->m; ++i) {
            double projected = project_interval(workspace->Ax[i], workspace->AL[i], workspace->AU[i]);
            double scale_fact = row_scale_at(scaling, i);
            workspace->Rp[i] = (projected - workspace->Ax[i]) * scale_fact;
            if (fabs(workspace->Ax[i] * scale_fact) > max_scaled_ax) {
                max_scaled_ax = fabs(workspace->Ax[i] * scale_fact);
            }
        }
        residuals->err_Rp_org_bar = vec_norm_inf(workspace->Rp, workspace->m) /
                                    (1.0 + max2(scaling->norm_b_org, max_scaled_ax));
    }

    if (iter == 0) {
        double bound_violation = compute_bound_violation(workspace, scaling);
        if (bound_violation > residuals->err_Rp_org_bar) {
            residuals->err_Rp_org_bar = bound_violation;
        }
    }

    residuals->KKTx_and_gap_org_bar = max3(residuals->err_Rp_org_bar,
                                           residuals->err_Rd_org_bar,
                                           residuals->rel_gap_bar);
    residuals->is_updated = 1;
    return HPRQP_STATUS_OK;
}