#include "shared/sparse.h"

#include <stdlib.h>
#include <string.h>

static void matrix_csr_reset(HPRQPMatrixCSR *matrix)
{
    if (matrix == NULL) {
        return;
    }

    matrix->nrow = 0;
    matrix->ncol = 0;
    matrix->nnz = 0;
    matrix->row_ptr = NULL;
    matrix->col_ind = NULL;
    matrix->values = NULL;
    matrix->owns_data = 0;
}

HPRQPStatus matrix_csr_alloc(HPRQPMatrixCSR *matrix, size_t nrow, size_t ncol, size_t nnz)
{
    if (matrix == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    matrix_csr_reset(matrix);
    matrix->row_ptr = (size_t *)calloc(nrow + 1, sizeof(size_t));
    matrix->col_ind = (size_t *)calloc(nnz > 0 ? nnz : 1, sizeof(size_t));
    matrix->values = (double *)calloc(nnz > 0 ? nnz : 1, sizeof(double));
    if (matrix->row_ptr == NULL || matrix->col_ind == NULL || matrix->values == NULL) {
        matrix_csr_free(matrix);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    matrix->nrow = nrow;
    matrix->ncol = ncol;
    matrix->nnz = nnz;
    matrix->owns_data = 1;
    return HPRQP_STATUS_OK;
}

void matrix_csr_free(HPRQPMatrixCSR *matrix)
{
    if (matrix == NULL) {
        return;
    }

    if (matrix->owns_data) {
        free(matrix->row_ptr);
        free(matrix->col_ind);
        free(matrix->values);
    }

    matrix_csr_reset(matrix);
}

HPRQPStatus matrix_csr_copy(HPRQPMatrixCSR *dest, const HPRQPMatrixCSR *src)
{
    if (dest == NULL || src == NULL || !matrix_csr_is_valid(src)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (dest == src) {
        return HPRQP_STATUS_OK;
    }

    {
        HPRQPStatus status = matrix_csr_alloc(dest, src->nrow, src->ncol, src->nnz);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    memcpy(dest->row_ptr, src->row_ptr, (src->nrow + 1) * sizeof(size_t));
    memcpy(dest->col_ind, src->col_ind, src->nnz * sizeof(size_t));
    memcpy(dest->values, src->values, src->nnz * sizeof(double));
    return HPRQP_STATUS_OK;
}

HPRQPStatus matrix_csr_transpose(HPRQPMatrixCSR *dest, const HPRQPMatrixCSR *src)
{
    size_t row;
    size_t idx;
    size_t *next;

    if (dest == NULL || src == NULL || !matrix_csr_is_valid(src)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    {
        HPRQPStatus status = matrix_csr_alloc(dest, src->ncol, src->nrow, src->nnz);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    for (idx = 0; idx < src->nnz; ++idx) {
        dest->row_ptr[src->col_ind[idx] + 1] += 1;
    }

    for (row = 0; row < dest->nrow; ++row) {
        dest->row_ptr[row + 1] += dest->row_ptr[row];
    }

    next = (size_t *)malloc(dest->nrow * sizeof(size_t));
    if (next == NULL) {
        matrix_csr_free(dest);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    memcpy(next, dest->row_ptr, dest->nrow * sizeof(size_t));

    for (row = 0; row < src->nrow; ++row) {
        for (idx = src->row_ptr[row]; idx < src->row_ptr[row + 1]; ++idx) {
            size_t col = src->col_ind[idx];
            size_t pos = next[col]++;
            dest->col_ind[pos] = row;
            dest->values[pos] = src->values[idx];
        }
    }

    free(next);
    return HPRQP_STATUS_OK;
}

HPRQPStatus matrix_csr_matvec(const HPRQPMatrixCSR *matrix, const double *x, double *y)
{
    size_t row;

    if (!matrix_csr_is_valid(matrix) || x == NULL || y == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    for (row = 0; row < matrix->nrow; ++row) {
        size_t idx;
        double sum = 0.0;
        for (idx = matrix->row_ptr[row]; idx < matrix->row_ptr[row + 1]; ++idx) {
            sum += matrix->values[idx] * x[matrix->col_ind[idx]];
        }
        y[row] = sum;
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus matrix_csr_transpose_matvec(const HPRQPMatrixCSR *matrix, const double *x, double *y)
{
    size_t row;

    if (!matrix_csr_is_valid(matrix) || x == NULL || y == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    for (row = 0; row < matrix->ncol; ++row) {
        y[row] = 0.0;
    }

    for (row = 0; row < matrix->nrow; ++row) {
        size_t idx;
        double x_row = x[row];
        for (idx = matrix->row_ptr[row]; idx < matrix->row_ptr[row + 1]; ++idx) {
            y[matrix->col_ind[idx]] += matrix->values[idx] * x_row;
        }
    }

    return HPRQP_STATUS_OK;
}

int matrix_csr_is_valid(const HPRQPMatrixCSR *matrix)
{
    if (matrix == NULL || matrix->row_ptr == NULL || matrix->col_ind == NULL || matrix->values == NULL) {
        return 0;
    }

    if (matrix->row_ptr[0] != 0 || matrix->row_ptr[matrix->nrow] != matrix->nnz) {
        return 0;
    }

    return 1;
}
