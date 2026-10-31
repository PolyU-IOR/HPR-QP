#include "cpu/scale.h"

#include "shared/sparse.h"
#include "shared/vector.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double max_abs_constraint_bounds(const double *AL, const double *AU, size_t m)
{
    size_t i;
    double norm_b = 0.0;
    for (i = 0; i < m; ++i) {
        double al = isinf(AL[i]) ? 0.0 : fabs(AL[i]);
        double au = isinf(AU[i]) ? 0.0 : fabs(AU[i]);
        double value = al > au ? al : au;
        if (value > norm_b) {
            norm_b = value;
        }
    }
    return norm_b;
}

static double norm2_constraint_bounds(const double *AL, const double *AU, size_t m)
{
    size_t i;
    double sum_sq = 0.0;

    for (i = 0; i < m; ++i) {
        double al = isinf(AL[i]) ? 0.0 : fabs(AL[i]);
        double au = isinf(AU[i]) ? 0.0 : fabs(AU[i]);
        double value = al > au ? al : au;
        sum_sq += value * value;
    }

    return sqrt(sum_sq);
}

static double *alloc_vec(size_t n)
{
    if (n == 0) {
        return NULL;
    }
    return (double *)malloc(n * sizeof(double));
}

static HPRQPStatus alloc_copy(double **dest, const double *src, size_t n)
{
    if (n == 0) {
        *dest = NULL;
        return HPRQP_STATUS_OK;
    }
    *dest = (double *)malloc(n * sizeof(double));
    if (*dest == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    memcpy(*dest, src, n * sizeof(double));
    return HPRQP_STATUS_OK;
}

HPRQPStatus curtis_reid_norms_compute(const HPRQPMatrixCSR *A,
                                      const HPRQPMatrixCSR *AT,
                                      size_t iterations,
                                      double *row_norm,
                                      double *col_norm)
{
    double *row_log_scale;
    double *col_log_scale;
    size_t iter;
    size_t row;

    if (A == NULL || AT == NULL ||
        (A->nrow > 0 && row_norm == NULL) ||
        (A->ncol > 0 && col_norm == NULL) ||
        A->nrow != AT->ncol || A->ncol != AT->nrow) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    row_log_scale = A->nrow > 0 ? (double *)calloc(A->nrow, sizeof(double)) : NULL;
    col_log_scale = A->ncol > 0 ? (double *)calloc(A->ncol, sizeof(double)) : NULL;
    if ((A->nrow > 0 && row_log_scale == NULL) ||
        (A->ncol > 0 && col_log_scale == NULL)) {
        free(row_log_scale);
        free(col_log_scale);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    for (iter = 0; iter < iterations; ++iter) {
        for (row = 0; row < A->nrow; ++row) {
            const size_t start = A->row_ptr[row];
            const size_t stop = A->row_ptr[row + 1];
            const size_t count = stop - start;
            double sum = 0.0;
            size_t idx;

            for (idx = start; idx < stop; ++idx) {
                const double magnitude = fmax(fabs(A->values[idx]), 1e-300);
                sum += -log(magnitude) - col_log_scale[A->col_ind[idx]];
            }
            row_log_scale[row] = count > 0 ? sum / (double)count : 0.0;
        }

        for (row = 0; row < AT->nrow; ++row) {
            const size_t start = AT->row_ptr[row];
            const size_t stop = AT->row_ptr[row + 1];
            const size_t count = stop - start;
            double sum = 0.0;
            size_t idx;

            for (idx = start; idx < stop; ++idx) {
                const double magnitude = fmax(fabs(AT->values[idx]), 1e-300);
                sum += -log(magnitude) - row_log_scale[AT->col_ind[idx]];
            }
            col_log_scale[row] = count > 0 ? sum / (double)count : 0.0;
        }
    }

    for (row = 0; row < A->nrow; ++row) {
        const double scale = fmin(fmax(exp(row_log_scale[row]), 1e-30), 1e30);
        row_norm[row] = 1.0 / scale;
    }
    for (row = 0; row < A->ncol; ++row) {
        const double scale = fmin(fmax(exp(col_log_scale[row]), 1e-30), 1e30);
        col_norm[row] = 1.0 / scale;
    }

    free(row_log_scale);
    free(col_log_scale);
    return HPRQP_STATUS_OK;
}

static void compute_col_max_abs(double *col_norm, const HPRQPMatrixCSR *A, const HPRQPMatrixCSR *Q)
{
    size_t i;
    size_t row;
    for (i = 0; i < A->ncol; ++i) {
        col_norm[i] = 0.0;
    }

    for (row = 0; row < A->nrow; ++row) {
        size_t idx;
        for (idx = A->row_ptr[row]; idx < A->row_ptr[row + 1]; ++idx) {
            size_t col = A->col_ind[idx];
            double value = fabs(A->values[idx]);
            if (value > col_norm[col]) {
                col_norm[col] = value;
            }
        }
    }

    for (row = 0; row < Q->nrow; ++row) {
        size_t idx;
        for (idx = Q->row_ptr[row]; idx < Q->row_ptr[row + 1]; ++idx) {
            size_t col = Q->col_ind[idx];
            double value = fabs(Q->values[idx]);
            if (value > col_norm[col]) {
                col_norm[col] = value;
            }
        }
    }

    for (i = 0; i < A->ncol; ++i) {
        col_norm[i] = sqrt(col_norm[i]);
        if (col_norm[i] == 0.0) {
            col_norm[i] = 1.0;
        }
    }
}

static void compute_row_max_abs(double *row_norm, const HPRQPMatrixCSR *A)
{
    size_t row;
    for (row = 0; row < A->nrow; ++row) {
        size_t idx;
        double max_abs = 0.0;
        for (idx = A->row_ptr[row]; idx < A->row_ptr[row + 1]; ++idx) {
            double value = fabs(A->values[idx]);
            if (value > max_abs) {
                max_abs = value;
            }
        }
        row_norm[row] = sqrt(max_abs);
        if (row_norm[row] == 0.0) {
            row_norm[row] = 1.0;
        }
    }
}

static void compute_col_sum_abs(double *col_norm, const HPRQPMatrixCSR *A, const HPRQPMatrixCSR *Q)
{
    size_t i;
    size_t row;
    for (i = 0; i < A->ncol; ++i) {
        col_norm[i] = 0.0;
    }

    for (row = 0; row < A->nrow; ++row) {
        size_t idx;
        for (idx = A->row_ptr[row]; idx < A->row_ptr[row + 1]; ++idx) {
            col_norm[A->col_ind[idx]] += fabs(A->values[idx]);
        }
    }

    for (row = 0; row < Q->nrow; ++row) {
        size_t idx;
        for (idx = Q->row_ptr[row]; idx < Q->row_ptr[row + 1]; ++idx) {
            col_norm[Q->col_ind[idx]] += fabs(Q->values[idx]);
        }
    }

    for (i = 0; i < A->ncol; ++i) {
        col_norm[i] = sqrt(col_norm[i]);
        if (col_norm[i] == 0.0) {
            col_norm[i] = 1.0;
        }
    }
}

static void compute_row_sum_abs(double *row_norm, const HPRQPMatrixCSR *A)
{
    size_t row;
    for (row = 0; row < A->nrow; ++row) {
        size_t idx;
        double sum_abs = 0.0;
        for (idx = A->row_ptr[row]; idx < A->row_ptr[row + 1]; ++idx) {
            sum_abs += fabs(A->values[idx]);
        }
        row_norm[row] = sqrt(sum_abs);
        if (row_norm[row] == 0.0) {
            row_norm[row] = 1.0;
        }
    }
}

static void scale_q_matrix(HPRQPMatrixCSR *Q, const double *col_norm)
{
    size_t row;
    for (row = 0; row < Q->nrow; ++row) {
        size_t idx;
        double row_scale = col_norm[row];
        for (idx = Q->row_ptr[row]; idx < Q->row_ptr[row + 1]; ++idx) {
            Q->values[idx] /= row_scale;
            Q->values[idx] /= col_norm[Q->col_ind[idx]];
        }
    }
}

static void scale_a_matrix(HPRQPMatrixCSR *A, const double *row_norm, const double *col_norm)
{
    size_t row;
    for (row = 0; row < A->nrow; ++row) {
        size_t idx;
        double row_scale = row_norm[row];
        for (idx = A->row_ptr[row]; idx < A->row_ptr[row + 1]; ++idx) {
            A->values[idx] /= row_scale;
            A->values[idx] /= col_norm[A->col_ind[idx]];
        }
    }
}

static void scale_problem_vectors(HPRQPModel *model, const double *row_norm, const double *col_norm)
{
    size_t i;
    for (i = 0; i < model->Q.ncol; ++i) {
        model->c[i] /= col_norm[i];
        model->l[i] *= col_norm[i];
        model->u[i] *= col_norm[i];
    }
    for (i = 0; i < model->A.nrow; ++i) {
        if (!isinf(model->AL[i])) {
            model->AL[i] /= row_norm[i];
        }
        if (!isinf(model->AU[i])) {
            model->AU[i] /= row_norm[i];
        }
    }
}

void scaling_init(HPRQPScaling *scaling)
{
    if (scaling == NULL) {
        return;
    }
    memset(scaling, 0, sizeof(*scaling));
    scaling->b_scale = 1.0;
    scaling->c_scale = 1.0;
}

void scaling_free(HPRQPScaling *scaling)
{
    if (scaling == NULL) {
        return;
    }
    free(scaling->row_norm);
    free(scaling->col_norm);
    free(scaling->l_org);
    free(scaling->u_org);
    scaling_init(scaling);
}

HPRQPStatus scale_problem(HPRQPModel *model, const HPRQPParams *params, HPRQPScaling *scaling)
{
    HPRQPStatus status;
    double *temp_row_norm;
    double *temp_col_norm;
    size_t i;

    if (model == NULL || params == NULL || scaling == NULL || !model->owns_data) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    scaling_free(scaling);
    status = alloc_copy(&scaling->l_org, model->l, model->Q.ncol);
    if (status != HPRQP_STATUS_OK) {
        scaling_free(scaling);
        return status;
    }
    status = alloc_copy(&scaling->u_org, model->u, model->Q.ncol);
    if (status != HPRQP_STATUS_OK) {
        scaling_free(scaling);
        return status;
    }

    scaling->row_norm = alloc_vec(model->A.nrow);
    scaling->col_norm = alloc_vec(model->Q.ncol);
    temp_row_norm = alloc_vec(model->A.nrow);
    temp_col_norm = alloc_vec(model->Q.ncol);
    if ((model->A.nrow > 0 && (scaling->row_norm == NULL || temp_row_norm == NULL)) ||
        (model->Q.ncol > 0 && (scaling->col_norm == NULL || temp_col_norm == NULL))) {
        free(temp_row_norm);
        free(temp_col_norm);
        scaling_free(scaling);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    for (i = 0; i < model->A.nrow; ++i) {
        scaling->row_norm[i] = 1.0;
    }
    for (i = 0; i < model->Q.ncol; ++i) {
        scaling->col_norm[i] = 1.0;
    }

    scaling->norm_b_org = max_abs_constraint_bounds(model->AL, model->AU, model->A.nrow);
    scaling->norm_c_org = vec_norm_inf(model->c, model->Q.ncol);

    if (params->use_scaling) {
        size_t iter;
        if (params->use_cr_scaling) {
            status = curtis_reid_norms_compute(&model->A,
                                               &model->AT,
                                               params->cr_iterations,
                                               temp_row_norm,
                                               temp_col_norm);
            if (status != HPRQP_STATUS_OK) {
                free(temp_row_norm);
                free(temp_col_norm);
                scaling_free(scaling);
                return status;
            }
            for (i = 0; i < model->A.nrow; ++i) {
                scaling->row_norm[i] *= temp_row_norm[i];
            }
            for (i = 0; i < model->Q.ncol; ++i) {
                scaling->col_norm[i] *= temp_col_norm[i];
            }
            scale_q_matrix(&model->Q, temp_col_norm);
            scale_a_matrix(&model->A, temp_row_norm, temp_col_norm);
            scale_problem_vectors(model, temp_row_norm, temp_col_norm);
        }
        for (iter = 0; iter < 10; ++iter) {
            compute_col_max_abs(temp_col_norm, &model->A, &model->Q);
            compute_row_max_abs(temp_row_norm, &model->A);
            for (i = 0; i < model->A.nrow; ++i) {
                scaling->row_norm[i] *= temp_row_norm[i];
            }
            for (i = 0; i < model->Q.ncol; ++i) {
                scaling->col_norm[i] *= temp_col_norm[i];
            }
            scale_q_matrix(&model->Q, temp_col_norm);
            scale_a_matrix(&model->A, temp_row_norm, temp_col_norm);
            scale_problem_vectors(model, temp_row_norm, temp_col_norm);
        }
        compute_col_sum_abs(temp_col_norm, &model->A, &model->Q);
        compute_row_sum_abs(temp_row_norm, &model->A);
        for (i = 0; i < model->A.nrow; ++i) {
            scaling->row_norm[i] *= temp_row_norm[i];
        }
        for (i = 0; i < model->Q.ncol; ++i) {
            scaling->col_norm[i] *= temp_col_norm[i];
        }
        scale_q_matrix(&model->Q, temp_col_norm);
        scale_a_matrix(&model->A, temp_row_norm, temp_col_norm);
        scale_problem_vectors(model, temp_row_norm, temp_col_norm);
    }

    free(temp_row_norm);
    free(temp_col_norm);

    matrix_csr_free(&model->AT);
    status = matrix_csr_transpose(&model->AT, &model->A);
    if (status != HPRQP_STATUS_OK) {
        scaling_free(scaling);
        return status;
    }

    scaling->norm_b = norm2_constraint_bounds(model->AL, model->AU, model->A.nrow);
    scaling->norm_c = vec_norm2(model->c, model->Q.ncol);
    scaling->b_scale = 1.0;
    scaling->c_scale = 1.0;
    return HPRQP_STATUS_OK;
}
