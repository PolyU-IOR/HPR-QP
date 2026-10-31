#include "shared/build.h"

#include "shared/sparse.h"

#include <math.h>

#include <stdlib.h>
#include <string.h>

typedef struct MatrixEntry {
    size_t row;
    size_t col;
    double value;
} MatrixEntry;

static int matrix_all_finite(const HPRQPMatrixCSR *matrix);

static void free_if_owned(double **data)
{
    if (data != NULL && *data != NULL) {
        free(*data);
        *data = NULL;
    }
}

static void free_size_t_if_owned(size_t **data)
{
    if (data != NULL && *data != NULL) {
        free(*data);
        *data = NULL;
    }
}

static HPRQPStatus alloc_and_copy(double **dest, const double *src, size_t n)
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

static HPRQPStatus alloc_and_copy_size_t(size_t **dest, const size_t *src, size_t n)
{
    if (n == 0) {
        *dest = NULL;
        return HPRQP_STATUS_OK;
    }
    *dest = (size_t *)malloc(n * sizeof(size_t));
    if (*dest == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    memcpy(*dest, src, n * sizeof(size_t));
    return HPRQP_STATUS_OK;
}

static void quadratic_constraints_free(HPRQPQuadraticConstraints *qc)
{
    if (qc == NULL) {
        return;
    }
    matrix_csr_free(&qc->linear);
    matrix_csr_free(&qc->diagonal);
    free_if_owned(&qc->offset);
    free_size_t_if_owned(&qc->cone_ptr);
    memset(qc, 0, sizeof(*qc));
}

static HPRQPStatus quadratic_constraints_copy(HPRQPQuadraticConstraints *dest,
                                               const HPRQPQuadraticConstraints *src)
{
    HPRQPStatus status;
    memset(dest, 0, sizeof(*dest));
    if (src->count == 0) {
        return HPRQP_STATUS_OK;
    }
    status = matrix_csr_copy(&dest->linear, &src->linear);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = matrix_csr_copy(&dest->diagonal, &src->diagonal);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = alloc_and_copy(&dest->offset, src->offset, src->count);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    status = alloc_and_copy_size_t(&dest->cone_ptr, src->cone_ptr,
                                   src->count + 1);
    if (status != HPRQP_STATUS_OK) {
        goto fail;
    }
    dest->count = src->count;
    dest->cone_dim = src->cone_dim;
    dest->disjoint_quadratic_support = src->disjoint_quadratic_support;
    return HPRQP_STATUS_OK;

fail:
    quadratic_constraints_free(dest);
    return status;
}

static int quadratic_constraints_are_valid(HPRQPQuadraticConstraints *qc,
                                            size_t n)
{
    size_t i;
    size_t idx;
    unsigned char *seen;

    if (qc->count == 0) {
        return qc->cone_dim == 0;
    }
    if (!matrix_csr_is_valid(&qc->linear) ||
        !matrix_csr_is_valid(&qc->diagonal) ||
        qc->linear.nrow != qc->count || qc->diagonal.nrow != qc->count ||
        qc->linear.ncol != n || qc->diagonal.ncol != n ||
        qc->offset == NULL || qc->cone_ptr == NULL ||
        qc->cone_ptr[0] != 0) {
        return 0;
    }
    for (i = 0; i < qc->count; ++i) {
        size_t expected = qc->cone_ptr[i] + 2 +
                          qc->diagonal.row_ptr[i + 1] -
                          qc->diagonal.row_ptr[i];
        if (!isfinite(qc->offset[i]) || qc->cone_ptr[i + 1] != expected) {
            return 0;
        }
    }
    if (qc->cone_ptr[qc->count] != qc->cone_dim ||
        !matrix_all_finite(&qc->linear) ||
        !matrix_all_finite(&qc->diagonal)) {
        return 0;
    }
    for (idx = 0; idx < qc->linear.nnz; ++idx) {
        if (qc->linear.col_ind[idx] >= n) {
            return 0;
        }
    }
    seen = (unsigned char *)calloc(n > 0 ? n : 1, sizeof(unsigned char));
    if (seen == NULL) {
        return 0;
    }
    qc->disjoint_quadratic_support = 1;
    for (idx = 0; idx < qc->diagonal.nnz; ++idx) {
        size_t col = qc->diagonal.col_ind[idx];
        if (col >= n || qc->diagonal.values[idx] < 0.0) {
            free(seen);
            return 0;
        }
        if (qc->diagonal.values[idx] > 0.0) {
            if (seen[col]) {
                qc->disjoint_quadratic_support = 0;
            }
            seen[col] = 1;
        }
    }
    free(seen);
    return 1;
}

static int matrix_entry_compare(const void *lhs, const void *rhs)
{
    const MatrixEntry *a = (const MatrixEntry *)lhs;
    const MatrixEntry *b = (const MatrixEntry *)rhs;
    if (a->row < b->row) {
        return -1;
    }
    if (a->row > b->row) {
        return 1;
    }
    if (a->col < b->col) {
        return -1;
    }
    if (a->col > b->col) {
        return 1;
    }
    return 0;
}

static int bounds_allow_infinity_only(const double *values, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        if (isnan(values[i])) {
            return 0;
        }
    }
    return 1;
}

static int vector_all_finite(const double *values, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static int matrix_all_finite(const HPRQPMatrixCSR *matrix)
{
    size_t idx;
    if (!matrix_csr_is_valid(matrix)) {
        return 0;
    }
    for (idx = 0; idx < matrix->nnz; ++idx) {
        if (!isfinite(matrix->values[idx])) {
            return 0;
        }
    }
    return 1;
}

static HPRQPStatus build_csr_from_entries(HPRQPMatrixCSR *matrix,
                                          size_t nrow,
                                          size_t ncol,
                                          MatrixEntry *entries,
                                          size_t nentries,
                                          double drop_tolerance)
{
    HPRQPStatus status;
    size_t idx;
    size_t unique_count;

    qsort(entries, nentries, sizeof(MatrixEntry), matrix_entry_compare);

    unique_count = 0;
    for (idx = 0; idx < nentries; ++idx) {
        if (fabs(entries[idx].value) <= drop_tolerance) {
            continue;
        }
        if (unique_count > 0 &&
            entries[idx].row == entries[unique_count - 1].row &&
            entries[idx].col == entries[unique_count - 1].col) {
            entries[unique_count - 1].value += entries[idx].value;
        } else {
            entries[unique_count] = entries[idx];
            unique_count += 1;
        }
    }

    for (idx = 0; idx < unique_count; ++idx) {
        if (fabs(entries[idx].value) <= drop_tolerance) {
            entries[idx].value = 0.0;
        }
    }

    {
        size_t kept = 0;
        for (idx = 0; idx < unique_count; ++idx) {
            if (entries[idx].value != 0.0) {
                entries[kept++] = entries[idx];
            }
        }
        unique_count = kept;
    }

    status = matrix_csr_alloc(matrix, nrow, ncol, unique_count);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    for (idx = 0; idx < unique_count; ++idx) {
        matrix->row_ptr[entries[idx].row + 1] += 1;
    }
    for (idx = 0; idx < nrow; ++idx) {
        matrix->row_ptr[idx + 1] += matrix->row_ptr[idx];
    }
    for (idx = 0; idx < unique_count; ++idx) {
        size_t pos = idx;
        matrix->col_ind[pos] = entries[idx].col;
        matrix->values[pos] = entries[idx].value;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus symmetrize_q_matrix(HPRQPMatrixCSR *dest, const HPRQPMatrixCSR *src, double factor)
{
    MatrixEntry *entries;
    size_t row;
    size_t idx;
    size_t nentries = 2 * src->nnz;
    HPRQPStatus status;

    entries = (MatrixEntry *)malloc((nentries > 0 ? nentries : 1) * sizeof(MatrixEntry));
    if (entries == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    nentries = 0;
    for (row = 0; row < src->nrow; ++row) {
        for (idx = src->row_ptr[row]; idx < src->row_ptr[row + 1]; ++idx) {
            entries[nentries].row = row;
            entries[nentries].col = src->col_ind[idx];
            entries[nentries].value = factor * src->values[idx];
            nentries += 1;

            entries[nentries].row = src->col_ind[idx];
            entries[nentries].col = row;
            entries[nentries].value = factor * src->values[idx];
            nentries += 1;
        }
    }

    status = build_csr_from_entries(dest, src->nrow, src->ncol, entries, nentries, 1e-15);
    free(entries);
    return status;
}

static HPRQPStatus delete_empty_and_free_rows(HPRQPModel *model)
{
    size_t row;
    size_t kept_rows;
    size_t new_nnz;
    size_t *keep;
    double *new_AL;
    double *new_AU;
    HPRQPMatrixCSR new_A;
    HPRQPStatus status;

    if (model->A.nrow == 0) {
        return HPRQP_STATUS_OK;
    }

    keep = (size_t *)malloc(model->A.nrow * sizeof(size_t));
    if (keep == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    kept_rows = 0;
    new_nnz = 0;
    for (row = 0; row < model->A.nrow; ++row) {
        size_t row_nnz = model->A.row_ptr[row + 1] - model->A.row_ptr[row];
        int unconstrained = isinf(model->AL[row]) && model->AL[row] < 0.0 &&
                            isinf(model->AU[row]) && model->AU[row] > 0.0;
        if (row_nnz > 0 && !unconstrained) {
            keep[kept_rows++] = row;
            new_nnz += row_nnz;
        }
    }

    if (kept_rows == model->A.nrow) {
        free(keep);
        return HPRQP_STATUS_OK;
    }

    new_AL = kept_rows > 0 ? (double *)malloc(kept_rows * sizeof(double)) : NULL;
    new_AU = kept_rows > 0 ? (double *)malloc(kept_rows * sizeof(double)) : NULL;
    if (kept_rows > 0 && (new_AL == NULL || new_AU == NULL)) {
        free(keep);
        free(new_AL);
        free(new_AU);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    status = matrix_csr_alloc(&new_A, kept_rows, model->A.ncol, new_nnz);
    if (status != HPRQP_STATUS_OK) {
        free(keep);
        free(new_AL);
        free(new_AU);
        return status;
    }

    new_A.row_ptr[0] = 0;
    new_nnz = 0;
    for (row = 0; row < kept_rows; ++row) {
        size_t old_row = keep[row];
        size_t idx2;
        new_AL[row] = model->AL[old_row];
        new_AU[row] = model->AU[old_row];
        for (idx2 = model->A.row_ptr[old_row]; idx2 < model->A.row_ptr[old_row + 1]; ++idx2) {
            new_A.col_ind[new_nnz] = model->A.col_ind[idx2];
            new_A.values[new_nnz] = model->A.values[idx2];
            new_nnz += 1;
        }
        new_A.row_ptr[row + 1] = new_nnz;
    }

    matrix_csr_free(&model->A);
    matrix_csr_free(&model->AT);
    free_if_owned(&model->AL);
    free_if_owned(&model->AU);

    model->A = new_A;
    model->AL = new_AL;
    model->AU = new_AU;
    status = matrix_csr_transpose(&model->AT, &model->A);
    free(keep);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus preprocess_model(HPRQPModel *model)
{
    HPRQPMatrixCSR sym_q;
    HPRQPStatus status;
    size_t idx;

    if (model == NULL || !model->owns_data) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (!matrix_csr_is_valid(&model->Q) || !matrix_csr_is_valid(&model->A) || !matrix_csr_is_valid(&model->AT)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (model->Q.nrow != model->Q.ncol || model->Q.nrow != model->A.ncol) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (model->A.nrow > 0 && (model->AL == NULL || model->AU == NULL)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (!matrix_all_finite(&model->Q) || !matrix_all_finite(&model->A) || !vector_all_finite(model->c, model->Q.ncol)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (!bounds_allow_infinity_only(model->AL, model->A.nrow) ||
        !bounds_allow_infinity_only(model->AU, model->A.nrow) ||
        !bounds_allow_infinity_only(model->l, model->Q.ncol) ||
        !bounds_allow_infinity_only(model->u, model->Q.ncol)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (!quadratic_constraints_are_valid(&model->qc, model->Q.ncol)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    for (idx = 0; idx < model->Q.ncol; ++idx) {
        if (model->l[idx] > model->u[idx]) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
    }
    for (idx = 0; idx < model->A.nrow; ++idx) {
        if (model->AL[idx] > model->AU[idx]) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
    }

    sym_q.nrow = 0;
    sym_q.ncol = 0;
    sym_q.nnz = 0;
    sym_q.row_ptr = NULL;
    sym_q.col_ind = NULL;
    sym_q.values = NULL;
    sym_q.owns_data = 0;

    status = symmetrize_q_matrix(&sym_q, &model->Q, 0.5);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    matrix_csr_free(&model->Q);
    model->Q = sym_q;

    return delete_empty_and_free_rows(model);
}

void model_init(HPRQPModel *model)
{
    if (model == NULL) {
        return;
    }

    memset(model, 0, sizeof(*model));
}

void model_free(HPRQPModel *model)
{
    if (model == NULL) {
        return;
    }

    if (model->owns_data) {
        matrix_csr_free(&model->Q);
        matrix_csr_free(&model->A);
        matrix_csr_free(&model->AT);
        free_if_owned(&model->c);
        free_if_owned(&model->AL);
        free_if_owned(&model->AU);
        free_if_owned(&model->l);
        free_if_owned(&model->u);
        quadratic_constraints_free(&model->qc);
    }

    model_init(model);
}

HPRQPStatus model_copy(HPRQPModel *dest, const HPRQPModel *src)
{
    HPRQPStatus status;

    if (dest == NULL || src == NULL || !src->owns_data) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model_free(dest);

    status = matrix_csr_copy(&dest->Q, &src->Q);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    status = matrix_csr_copy(&dest->A, &src->A);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    status = matrix_csr_copy(&dest->AT, &src->AT);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    status = alloc_and_copy(&dest->c, src->c, src->Q.ncol);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    status = alloc_and_copy(&dest->l, src->l, src->Q.ncol);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    status = alloc_and_copy(&dest->u, src->u, src->Q.ncol);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    status = alloc_and_copy(&dest->AL, src->AL, src->A.nrow);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    status = alloc_and_copy(&dest->AU, src->AU, src->A.nrow);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    status = quadratic_constraints_copy(&dest->qc, &src->qc);
    if (status != HPRQP_STATUS_OK) {
        model_free(dest);
        return status;
    }

    dest->obj_constant = src->obj_constant;
    dest->owns_data = 1;
    return HPRQP_STATUS_OK;
}

HPRQPStatus build_model_from_arrays(const HPRQPMatrixCSR *Q,
                                    const HPRQPMatrixCSR *A,
                                    const double *c,
                                    const double *AL,
                                    const double *AU,
                                    const double *l,
                                    const double *u,
                                    double obj_constant,
                                    HPRQPModel *model)
{
    HPRQPStatus status;
    size_t m;
    size_t n;

    if (Q == NULL || A == NULL || model == NULL || c == NULL || l == NULL || u == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (!matrix_csr_is_valid(Q) || !matrix_csr_is_valid(A)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (Q->nrow != Q->ncol || Q->nrow != A->ncol) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    m = A->nrow;
    n = A->ncol;

    if (m > 0 && (AL == NULL || AU == NULL)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model_free(model);

    status = matrix_csr_copy(&model->Q, Q);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }

    status = matrix_csr_copy(&model->A, A);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }

    status = matrix_csr_transpose(&model->AT, A);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }

    status = alloc_and_copy(&model->c, c, n);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }

    status = alloc_and_copy(&model->l, l, n);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }

    status = alloc_and_copy(&model->u, u, n);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }

    status = alloc_and_copy(&model->AL, AL, m);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }

    status = alloc_and_copy(&model->AU, AU, m);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }

    model->obj_constant = obj_constant;
    model->owns_data = 1;
    return preprocess_model(model);
}

HPRQPStatus build_from_QAbc(const HPRQPMatrixCSR *Q,
                            const HPRQPMatrixCSR *A,
                            const double *c,
                            const double *AL,
                            const double *AU,
                            const double *l,
                            const double *u,
                            double obj_constant,
                            HPRQPModel *model)
{
    return build_model_from_arrays(Q, A, c, AL, AU, l, u, obj_constant, model);
}
