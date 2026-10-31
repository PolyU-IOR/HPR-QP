#include "cpu/workspace.h"

#include "cpu/eigs.h"
#include "shared/parallel.h"
#include "shared/vector.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double *alloc_vector(size_t n)
{
    if (n == 0) {
        return NULL;
    }
    return (double *)calloc(n, sizeof(double));
}

static void free_vector(double **x)
{
    if (x != NULL && *x != NULL) {
        free(*x);
        *x = NULL;
    }
}

static int q_is_diagonal(const HPRQPMatrixCSR *Q, double *diag_Q)
{
    size_t row;
    int is_diag = 1;
    for (row = 0; row < Q->nrow; ++row) {
        size_t idx;
        diag_Q[row] = 0.0;
        for (idx = Q->row_ptr[row]; idx < Q->row_ptr[row + 1]; ++idx) {
            size_t col = Q->col_ind[idx];
            if (col == row) {
                diag_Q[row] += Q->values[idx];
            } else if (fabs(Q->values[idx]) > 1e-15) {
                is_diag = 0;
            }
        }
    }
    return is_diag;
}

static int detect_noC(const HPRQPModel *model)
{
    size_t i;
    size_t free_count = 0;

    if (model == NULL || model->Q.ncol == 0) {
        return 0;
    }

    for (i = 0; i < model->Q.ncol; ++i) {
        if (isinf(model->l[i]) && isinf(model->u[i])) {
            free_count += 1;
        }
    }

    return 5 * free_count > 4 * model->Q.ncol;
}

static double compute_initial_sigma(const HPRQPModel *model)
{
    {
        size_t i;
        double norm_b_sq = 0.0;
        double norm_c = vec_norm2(model->c, model->Q.ncol);
        for (i = 0; i < model->A.nrow; ++i) {
            double al = isinf(model->AL[i]) ? 0.0 : fabs(model->AL[i]);
            double au = isinf(model->AU[i]) ? 0.0 : fabs(model->AU[i]);
            double value = al > au ? al : au;
            norm_b_sq += value * value;
        }
        {
            double norm_b = sqrt(norm_b_sq);
        if (norm_b > 1e-16 && norm_c > 1e-16 && norm_b < 1e16 && norm_c < 1e16) {
            return norm_b / norm_c;
        }
        }
    }

    return 1.0;
}

static HPRQPStatus allocate_work_vector_group(HPRQPWorkspace *ws)
{
    ws->x = alloc_vector(ws->n);
    ws->x_bar = alloc_vector(ws->n);
    ws->x_hat = alloc_vector(ws->n);
    ws->w = alloc_vector(ws->n);
    ws->w_bar = alloc_vector(ws->n);
    ws->w_hat = alloc_vector(ws->n);
    ws->y = alloc_vector(ws->m);
    ws->y_bar = alloc_vector(ws->m);
    ws->y_hat = alloc_vector(ws->m);
    ws->Ax = alloc_vector(ws->m);
    ws->ATy = alloc_vector(ws->n);
    ws->ATy_bar = alloc_vector(ws->n);
    ws->ATdy = alloc_vector(ws->n);
    ws->Qw = alloc_vector(ws->n);
    ws->Qw_bar = alloc_vector(ws->n);
    ws->Qx = alloc_vector(ws->n);
    ws->dQw = alloc_vector(ws->n);
    ws->QATdy = alloc_vector(ws->n);
    ws->dx = alloc_vector(ws->n);
    ws->dy = alloc_vector(ws->m);
    ws->dw = alloc_vector(ws->n);
    ws->Rp = alloc_vector(ws->m);
    ws->Rd = alloc_vector(ws->n);
    ws->s = alloc_vector(ws->m);
    ws->z_bar = alloc_vector(ws->n);
    ws->tempv = alloc_vector(ws->n);
    ws->diag_Q = alloc_vector(ws->n);
    ws->fact1 = alloc_vector(ws->n);
    ws->fact2 = alloc_vector(ws->n);
    ws->fact = alloc_vector(ws->n);
    ws->fact_M = alloc_vector(ws->n);
    ws->last_x = alloc_vector(ws->n);
    ws->last_y = alloc_vector(ws->m);
    ws->last_Qw = alloc_vector(ws->n);
    ws->last_w = alloc_vector(ws->n);
    ws->last_ATy = alloc_vector(ws->n);

    if (ws->x == NULL || ws->x_bar == NULL || ws->x_hat == NULL || ws->w == NULL || ws->w_bar == NULL || ws->w_hat == NULL ||
        (ws->m > 0 && (ws->y == NULL || ws->y_bar == NULL || ws->y_hat == NULL || ws->Ax == NULL || ws->dy == NULL || ws->Rp == NULL || ws->s == NULL)) ||
        ws->ATy == NULL || ws->ATy_bar == NULL || ws->ATdy == NULL || ws->Qw == NULL || ws->Qw_bar == NULL || ws->Qx == NULL ||
        ws->dQw == NULL || ws->QATdy == NULL || ws->dx == NULL || ws->dw == NULL || ws->Rd == NULL || ws->z_bar == NULL ||
        ws->tempv == NULL || ws->diag_Q == NULL || ws->fact1 == NULL || ws->fact2 == NULL || ws->fact == NULL || ws->fact_M == NULL ||
        ws->last_x == NULL || ws->last_Qw == NULL || ws->last_w == NULL || ws->last_ATy == NULL || (ws->m > 0 && ws->last_y == NULL)) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    return HPRQP_STATUS_OK;
}

void workspace_init(HPRQPWorkspace *workspace)
{
    if (workspace == NULL) {
        return;
    }
    memset(workspace, 0, sizeof(*workspace));
}

void workspace_free(HPRQPWorkspace *workspace)
{
    if (workspace == NULL) {
        return;
    }

    free_vector(&workspace->x);
    free_vector(&workspace->x_bar);
    free_vector(&workspace->x_hat);
    free_vector(&workspace->w);
    free_vector(&workspace->w_bar);
    free_vector(&workspace->w_hat);
    free_vector(&workspace->y);
    free_vector(&workspace->y_bar);
    free_vector(&workspace->y_hat);
    free_vector(&workspace->Ax);
    free_vector(&workspace->ATy);
    free_vector(&workspace->ATy_bar);
    free_vector(&workspace->ATdy);
    free_vector(&workspace->Qw);
    free_vector(&workspace->Qw_bar);
    free_vector(&workspace->Qx);
    free_vector(&workspace->dQw);
    free_vector(&workspace->QATdy);
    free_vector(&workspace->dx);
    free_vector(&workspace->dy);
    free_vector(&workspace->dw);
    free_vector(&workspace->Rp);
    free_vector(&workspace->Rd);
    free_vector(&workspace->s);
    free_vector(&workspace->z_bar);
    free_vector(&workspace->tempv);
    free_vector(&workspace->diag_Q);
    free_vector(&workspace->fact1);
    free_vector(&workspace->fact2);
    free_vector(&workspace->fact);
    free_vector(&workspace->fact_M);
    free_vector(&workspace->last_x);
    free_vector(&workspace->last_y);
    free_vector(&workspace->last_Qw);
    free_vector(&workspace->last_w);
    free_vector(&workspace->last_ATy);

    workspace_init(workspace);
}

HPRQPStatus workspace_allocate(const HPRQPModel *model,
                               const HPRQPParams *params,
                               HPRQPWorkspace *workspace)
{
    size_t i;
    HPRQPStatus status;

    if (model == NULL || params == NULL || workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    workspace_free(workspace);
    workspace->m = model->A.nrow;
    workspace->n = model->Q.ncol;
    workspace->Q = &model->Q;
    workspace->A = &model->A;
    workspace->AT = &model->AT;
    workspace->c = model->c;
    workspace->AL = model->AL;
    workspace->AU = model->AU;
    workspace->l = model->l;
    workspace->u = model->u;
    workspace->lambda_max_A = 0.0;
    workspace->lambda_max_Q = 0.0;
    workspace->method = params->method;
    workspace->parallel_majorizer_mode = params->parallel_majorizer_mode;
    workspace->metric_negative_tolerance = params->metric_negative_tolerance;
    workspace->to_check = 1;
    workspace->noC = detect_noC(model);
    workspace->sigma = params->initial_sigma > 0.0
                           ? params->initial_sigma
                           : compute_initial_sigma(model);
    if (workspace->sigma <= 0.0) {
        workspace_free(workspace);
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = allocate_work_vector_group(workspace);
    if (status != HPRQP_STATUS_OK) {
        workspace_free(workspace);
        return status;
    }

    workspace->Q_is_diag = q_is_diagonal(&model->Q, workspace->diag_Q);
    if (workspace->Q_is_diag) {
        for (i = 0; i < workspace->n; ++i) {
            double temp = 1.0 + workspace->sigma * workspace->diag_Q[i];
            double inv = 1.0 / temp;
            workspace->fact2[i] = inv;
            workspace->fact[i] = workspace->sigma * inv;
            workspace->fact1[i] = workspace->sigma * workspace->diag_Q[i] * inv;
            workspace->fact_M[i] = workspace->sigma * workspace->sigma * inv;
        }
    }

    status = estimate_eigenvalues(workspace, params);
    if (status != HPRQP_STATUS_OK) {
        workspace_free(workspace);
        return status;
    }
    workspace->eigenvalues_initialized = 1;

    if (workspace->method == HPRQP_METHOD_PARALLEL) {
        workspace->tau = parallel_select_tau(params,
                                             workspace->lambda_max_A,
                                             workspace->lambda_max_Q);
        workspace->tau_initial = workspace->tau;
        status = parallel_compute_coefficients(
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
        if (status != HPRQP_STATUS_OK) {
            workspace_free(workspace);
            return status;
        }
    } else {
        workspace->tau = NAN;
        workspace->tau_initial = NAN;
        workspace->d_y = NAN;
        workspace->d_w = NAN;
        workspace->parallel_y_rho = NAN;
        workspace->parallel_y_inv = NAN;
        workspace->parallel_w_gamma = NAN;
        workspace->parallel_w_inv = NAN;
    }

    return HPRQP_STATUS_OK;
}
