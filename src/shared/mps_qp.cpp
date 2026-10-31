#include "shared/build.h"
#include "shared/mps_reader.h"
#include "shared/sparse.h"

#include <cstddef>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cmath>

static HPRQPStatus copy_csr_matrix(HPRQPMatrixCSR *dest, const CSRMatrix *src)
{
    HPRQPStatus status;
    int row;
    int idx;

    if (dest == nullptr || src == nullptr) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = matrix_csr_alloc(dest,
                              static_cast<size_t>(src->nrows),
                              static_cast<size_t>(src->ncols),
                              static_cast<size_t>(src->nnz));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    for (row = 0; row <= src->nrows; ++row) {
        dest->row_ptr[row] = static_cast<size_t>(src->row_ptr[row]);
    }
    for (idx = 0; idx < src->nnz; ++idx) {
        dest->col_ind[idx] = static_cast<size_t>(src->col_idx[idx]);
        dest->values[idx] = src->values[idx];
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus build_constraint_matrix_direct(HPRQPMatrixCSR *dest,
                                                  const QPSData *qps,
                                                  bool *used_direct_path)
{
    HPRQPStatus status;
    size_t *next = nullptr;
    bool columns_ordered = true;
    int idx;
    int row;

    if (dest == nullptr || qps == nullptr || used_direct_path == nullptr ||
        qps->ncon < 0 || qps->nvar < 0 || qps->annz < 0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    *used_direct_path = false;
    for (idx = 0; idx < qps->annz; ++idx) {
        if (qps->arows[idx] < 0 || qps->arows[idx] >= qps->ncon ||
            qps->acols[idx] < 0 || qps->acols[idx] >= qps->nvar) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        if (idx > 0 && qps->acols[idx] < qps->acols[idx - 1]) {
            columns_ordered = false;
        }
    }

    if (!columns_ordered) {
        CSRMatrix *fallback = qpsdata_get_csr_matrix(qps);
        if (fallback == nullptr) {
            return HPRQP_STATUS_ALLOCATION_FAILED;
        }
        status = copy_csr_matrix(dest, fallback);
        csr_free(fallback);
        return status;
    }

    status = matrix_csr_alloc(dest,
                              static_cast<size_t>(qps->ncon),
                              static_cast<size_t>(qps->nvar),
                              static_cast<size_t>(qps->annz));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (qps->ncon == 0) {
        if (qps->annz != 0) {
            matrix_csr_free(dest);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        *used_direct_path = true;
        return HPRQP_STATUS_OK;
    }

    next = static_cast<size_t *>(malloc(static_cast<size_t>(qps->ncon) * sizeof(size_t)));
    if (next == nullptr) {
        matrix_csr_free(dest);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    for (idx = 0; idx < qps->annz; ++idx) {
        dest->row_ptr[static_cast<size_t>(qps->arows[idx]) + 1U] += 1U;
    }
    for (row = 0; row < qps->ncon; ++row) {
        dest->row_ptr[static_cast<size_t>(row) + 1U] +=
            dest->row_ptr[static_cast<size_t>(row)];
        next[row] = dest->row_ptr[static_cast<size_t>(row)];
    }

    /*
     * MPS COLUMNS records arrive in nondecreasing column order.  Scattering
     * them by row therefore produces the final CSR column order directly,
     * without a triplet sort or an intermediate int-indexed CSR matrix.
     */
    for (idx = 0; idx < qps->annz; ++idx) {
        const int input_row = qps->arows[idx];
        const size_t position = next[input_row]++;
        dest->col_ind[position] = static_cast<size_t>(qps->acols[idx]);
        dest->values[position] = qps->avals[idx];
    }

    {
        size_t input_start = 0;
        size_t output = 0;

        for (row = 0; row < qps->ncon; ++row) {
            const size_t input_end = next[row];
            size_t previous_col = 0;
            size_t input;

            dest->row_ptr[static_cast<size_t>(row)] = output;
            for (input = input_start; input < input_end; ++input) {
                if (input == input_start || dest->col_ind[input] != previous_col) {
                    previous_col = dest->col_ind[input];
                    dest->col_ind[output] = previous_col;
                    dest->values[output] = dest->values[input];
                    output += 1U;
                } else {
                    dest->values[output - 1U] += dest->values[input];
                }
            }
            input_start = input_end;
        }
        dest->row_ptr[static_cast<size_t>(qps->ncon)] = output;
        dest->nnz = output;
    }

    free(next);
    *used_direct_path = true;
    return HPRQP_STATUS_OK;
}

static void compress_ordered_csr(HPRQPMatrixCSR *matrix, size_t *row_ends)
{
    size_t input_start = 0;
    size_t output = 0;
    for (size_t row = 0; row < matrix->nrow; ++row) {
        const size_t input_end = row_ends[row];
        size_t previous_col = 0;
        matrix->row_ptr[row] = output;
        for (size_t input = input_start; input < input_end; ++input) {
            if (input == input_start || matrix->col_ind[input] != previous_col) {
                previous_col = matrix->col_ind[input];
                matrix->col_ind[output] = previous_col;
                matrix->values[output] = matrix->values[input];
                ++output;
            } else {
                matrix->values[output - 1] += matrix->values[input];
            }
        }
        input_start = input_end;
    }
    matrix->row_ptr[matrix->nrow] = output;
    matrix->nnz = output;
}

/*
 * Canonicalize every convex quadratic row to
 *
 *     0.5 x' Q_i x + c_i' x + d_i <= 0,   Q_i >= 0.
 *
 * MPS benchmark files use both an upper row with a positive-semidefinite
 * QCMATRIX and the equivalent lower row with a negative-semidefinite
 * QCMATRIX.  The latter needs one global sign change for the complete row,
 * including its linear coefficients and right-hand side.
 */
static HPRQPStatus build_qc_canonical_signs(double **signs_out,
                                             const QPSData *qps)
{
    double *signs;

    if (signs_out == nullptr || qps == nullptr || qps->nqc <= 0 ||
        qps->nqc > qps->ncon) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *signs_out = nullptr;
    signs = static_cast<double *>(
        malloc(static_cast<size_t>(qps->nqc) * sizeof(double)));
    if (signs == nullptr) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    for (int qc = 0; qc < qps->nqc; ++qc) {
        const int row = qps->qc_con_index[qc];
        if (row < 0 || row >= qps->ncon || qps->con_to_qc[row] != qc) {
            free(signs);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        if (qps->contypes[row] == RTYPE_LESSTHAN &&
            std::isinf(qps->lcon[row]) && qps->lcon[row] < 0.0 &&
            std::isfinite(qps->ucon[row])) {
            signs[qc] = 1.0;
        } else if (qps->contypes[row] == RTYPE_GREATERTHAN &&
                   std::isinf(qps->ucon[row]) && qps->ucon[row] > 0.0 &&
                   std::isfinite(qps->lcon[row])) {
            signs[qc] = -1.0;
        } else {
            free(signs);
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
    }

    for (int idx = 0; idx < qps->qcnnz; ++idx) {
        const int qc = qps->qcids[idx];
        if (qc < 0 || qc >= qps->nqc ||
            qps->qcrows[idx] < 0 || qps->qcrows[idx] >= qps->nvar ||
            qps->qccols[idx] != qps->qcrows[idx] ||
            !std::isfinite(qps->qcvals[idx]) ||
            signs[qc] * qps->qcvals[idx] < 0.0) {
            free(signs);
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
    }

    *signs_out = signs;
    return HPRQP_STATUS_OK;
}

/* Split the original MPS rows into ordinary affine rows and QC linear rows. */
static HPRQPStatus build_qcqp_constraint_matrices(HPRQPMatrixCSR *A,
                                                  HPRQPMatrixCSR *qc_linear,
                                                  const QPSData *qps,
                                                  const double *qc_signs)
{
    size_t ordinary_rows;
    size_t ordinary_nnz = 0;
    size_t qc_nnz = 0;
    size_t *next_A = nullptr;
    size_t *next_qc = nullptr;
    int *ordinary_row = nullptr;
    HPRQPStatus status;

    if (A == nullptr || qc_linear == nullptr || qps == nullptr ||
        qc_signs == nullptr ||
        qps->nqc <= 0 || qps->nqc > qps->ncon) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    ordinary_rows = static_cast<size_t>(qps->ncon - qps->nqc);
    ordinary_row = static_cast<int *>(
        malloc(static_cast<size_t>(qps->ncon) * sizeof(int)));
    if (ordinary_row == nullptr) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    int next_row = 0;
    for (int row = 0; row < qps->ncon; ++row) {
        ordinary_row[row] = qps->con_to_qc[row] >= 0 ? -1 : next_row++;
    }

    for (int idx = 0; idx < qps->annz; ++idx) {
        const int old_row = qps->arows[idx];
        if (old_row < 0 || old_row >= qps->ncon ||
            qps->acols[idx] < 0 || qps->acols[idx] >= qps->nvar) {
            free(ordinary_row);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        if (qps->con_to_qc[old_row] >= 0) {
            ++qc_nnz;
        } else {
            ++ordinary_nnz;
        }
    }

    status = matrix_csr_alloc(A, ordinary_rows,
                              static_cast<size_t>(qps->nvar), ordinary_nnz);
    if (status != HPRQP_STATUS_OK) {
        free(ordinary_row);
        return status;
    }
    status = matrix_csr_alloc(qc_linear, static_cast<size_t>(qps->nqc),
                              static_cast<size_t>(qps->nvar), qc_nnz);
    if (status != HPRQP_STATUS_OK) {
        matrix_csr_free(A);
        free(ordinary_row);
        return status;
    }

    for (int idx = 0; idx < qps->annz; ++idx) {
        const int old_row = qps->arows[idx];
        const int qc = qps->con_to_qc[old_row];
        if (qc >= 0) {
            ++qc_linear->row_ptr[static_cast<size_t>(qc) + 1];
        } else {
            ++A->row_ptr[static_cast<size_t>(ordinary_row[old_row]) + 1];
        }
    }
    for (size_t row = 0; row < A->nrow; ++row) {
        A->row_ptr[row + 1] += A->row_ptr[row];
    }
    for (size_t row = 0; row < qc_linear->nrow; ++row) {
        qc_linear->row_ptr[row + 1] += qc_linear->row_ptr[row];
    }

    next_A = static_cast<size_t *>(
        malloc((A->nrow > 0 ? A->nrow : 1) * sizeof(size_t)));
    next_qc = static_cast<size_t *>(
        malloc(qc_linear->nrow * sizeof(size_t)));
    if (next_A == nullptr || next_qc == nullptr) {
        free(next_A);
        free(next_qc);
        free(ordinary_row);
        matrix_csr_free(A);
        matrix_csr_free(qc_linear);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    for (size_t row = 0; row < A->nrow; ++row) next_A[row] = A->row_ptr[row];
    for (size_t row = 0; row < qc_linear->nrow; ++row) {
        next_qc[row] = qc_linear->row_ptr[row];
    }

    for (int idx = 0; idx < qps->annz; ++idx) {
        const int old_row = qps->arows[idx];
        const int qc = qps->con_to_qc[old_row];
        HPRQPMatrixCSR *dest = qc >= 0 ? qc_linear : A;
        const size_t row = qc >= 0 ? static_cast<size_t>(qc)
                                   : static_cast<size_t>(ordinary_row[old_row]);
        size_t *next = qc >= 0 ? next_qc : next_A;
        const size_t pos = next[row]++;
        dest->col_ind[pos] = static_cast<size_t>(qps->acols[idx]);
        dest->values[pos] = qps->avals[idx] *
                            (qc >= 0 ? qc_signs[qc] : 1.0);
    }

    /* COLUMNS order makes each scattered row ordered, so duplicates are adjacent. */
    compress_ordered_csr(A, next_A);
    compress_ordered_csr(qc_linear, next_qc);

    free(next_A);
    free(next_qc);
    free(ordinary_row);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus build_qc_diagonal(HPRQPMatrixCSR *dest,
                                     const QPSData *qps,
                                     const double *qc_signs)
{
    CSRMatrix *csr;
    if (dest == nullptr || qps == nullptr || qc_signs == nullptr ||
        qps->nqc <= 0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    for (int idx = 0; idx < qps->qcnnz; ++idx) {
        if (qps->qcids[idx] < 0 || qps->qcids[idx] >= qps->nqc ||
            qps->qcrows[idx] < 0 || qps->qcrows[idx] >= qps->nvar ||
            qps->qccols[idx] != qps->qcrows[idx] ||
            !std::isfinite(qps->qcvals[idx]) ||
            qc_signs[qps->qcids[idx]] * qps->qcvals[idx] < 0.0) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
    }
    csr = coo_to_csr(qps->nqc, qps->nvar, qps->qcnnz,
                     qps->qcids, qps->qcrows, qps->qcvals);
    if (csr == nullptr) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    HPRQPStatus status = copy_csr_matrix(dest, csr);
    csr_free(csr);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    /*
     * In the extended MPS QCMATRIX section, a diagonal record
     *
     *     Xj  Xj  a
     *
     * contributes a * x_j^2 to the constraint polynomial.  The solver's
     * canonical QCQP model stores Q_i in
     *
     *     0.5 * x' Q_i x + c_i' x + d_i <= 0,
     *
     * so its diagonal Hessian entry is 2 * a.  Copying the MPS value
     * directly made every quadratic constraint two times too weak.
     * Off-diagonal QCMATRIX entries are not supported by this path.
     */
    for (size_t row = 0; row < dest->nrow; ++row) {
        for (size_t idx = dest->row_ptr[row]; idx < dest->row_ptr[row + 1];
             ++idx) {
            dest->values[idx] *= 2.0 * qc_signs[row];
            if (!std::isfinite(dest->values[idx])) {
                matrix_csr_free(dest);
                return HPRQP_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return status;
}

static HPRQPStatus adopt_mps_model(HPRQPMatrixCSR *Q,
                                   HPRQPMatrixCSR *A,
                                   QPSData *qps,
                                   HPRQPModel *model)
{
    HPRQPStatus status;

    if (Q == nullptr || A == nullptr || qps == nullptr || model == nullptr ||
        !matrix_csr_is_valid(Q) || !matrix_csr_is_valid(A) ||
        Q->nrow != Q->ncol || Q->nrow != A->ncol) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model_free(model);
    model->Q = *Q;
    *Q = {};
    model->A = *A;
    *A = {};

    model->c = qps->c;
    qps->c = nullptr;
    model->AL = qps->lcon;
    qps->lcon = nullptr;
    model->AU = qps->ucon;
    qps->ucon = nullptr;
    model->l = qps->lvar;
    qps->lvar = nullptr;
    model->u = qps->uvar;
    qps->uvar = nullptr;
    model->obj_constant = qps->c0;
    model->owns_data = 1;

    status = matrix_csr_transpose(&model->AT, &model->A);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    return preprocess_model(model);
}

static HPRQPStatus adopt_mps_qcqp_model(HPRQPMatrixCSR *Q,
                                        HPRQPMatrixCSR *A,
                                        HPRQPMatrixCSR *qc_linear,
                                        HPRQPMatrixCSR *qc_diagonal,
                                        QPSData *qps,
                                        const double *qc_signs,
                                        HPRQPModel *model)
{
    HPRQPStatus status;
    size_t ordinary_row = 0;

    if (Q == nullptr || A == nullptr || qc_linear == nullptr ||
        qc_diagonal == nullptr || qps == nullptr || qc_signs == nullptr ||
        model == nullptr ||
        qps->nqc <= 0 || !matrix_csr_is_valid(Q) ||
        !matrix_csr_is_valid(A) || !matrix_csr_is_valid(qc_linear) ||
        !matrix_csr_is_valid(qc_diagonal) || Q->nrow != Q->ncol ||
        Q->nrow != A->ncol || Q->nrow != qc_linear->ncol ||
        Q->nrow != qc_diagonal->ncol ||
        qc_linear->nrow != static_cast<size_t>(qps->nqc) ||
        qc_diagonal->nrow != static_cast<size_t>(qps->nqc)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model_free(model);
    model->Q = *Q;
    *Q = {};
    model->A = *A;
    *A = {};
    model->qc.linear = *qc_linear;
    *qc_linear = {};
    model->qc.diagonal = *qc_diagonal;
    *qc_diagonal = {};
    model->qc.count = static_cast<size_t>(qps->nqc);

    model->c = qps->c;
    qps->c = nullptr;
    model->l = qps->lvar;
    qps->lvar = nullptr;
    model->u = qps->uvar;
    qps->uvar = nullptr;
    model->obj_constant = qps->c0;
    model->owns_data = 1;

    model->AL = model->A.nrow > 0
                    ? static_cast<double *>(malloc(model->A.nrow * sizeof(double)))
                    : nullptr;
    model->AU = model->A.nrow > 0
                    ? static_cast<double *>(malloc(model->A.nrow * sizeof(double)))
                    : nullptr;
    model->qc.offset = static_cast<double *>(
        malloc(model->qc.count * sizeof(double)));
    model->qc.cone_ptr = static_cast<size_t *>(
        malloc((model->qc.count + 1) * sizeof(size_t)));
    if ((model->A.nrow > 0 && (model->AL == nullptr || model->AU == nullptr)) ||
        model->qc.offset == nullptr || model->qc.cone_ptr == nullptr) {
        model_free(model);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    for (int old_row = 0; old_row < qps->ncon; ++old_row) {
        const int qc = qps->con_to_qc[old_row];
        if (qc >= 0) {
            if (qc_signs[qc] == 1.0) {
                model->qc.offset[static_cast<size_t>(qc)] =
                    -qps->ucon[old_row];
            } else if (qc_signs[qc] == -1.0) {
                model->qc.offset[static_cast<size_t>(qc)] =
                    qps->lcon[old_row];
            } else {
                model_free(model);
                return HPRQP_STATUS_INVALID_ARGUMENT;
            }
        } else {
            model->AL[ordinary_row] = qps->lcon[old_row];
            model->AU[ordinary_row] = qps->ucon[old_row];
            ++ordinary_row;
        }
    }
    if (ordinary_row != model->A.nrow) {
        model_free(model);
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model->qc.cone_ptr[0] = 0;
    for (size_t qc = 0; qc < model->qc.count; ++qc) {
        model->qc.cone_ptr[qc + 1] =
            model->qc.cone_ptr[qc] + 2 +
            model->qc.diagonal.row_ptr[qc + 1] -
            model->qc.diagonal.row_ptr[qc];
    }
    model->qc.cone_dim = model->qc.cone_ptr[model->qc.count];

    status = matrix_csr_transpose(&model->AT, &model->A);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
        return status;
    }
    status = preprocess_model(model);
    if (status != HPRQP_STATUS_OK) {
        model_free(model);
    }
    return status;
}

typedef struct MpsSymEntry {
    size_t row;
    size_t col;
    double value;
    int orientation;
} MpsSymEntry;

static int compare_mps_sym_entry(const void *lhs, const void *rhs)
{
    const MpsSymEntry *a = static_cast<const MpsSymEntry *>(lhs);
    const MpsSymEntry *b = static_cast<const MpsSymEntry *>(rhs);
    if (a->row < b->row) return -1;
    if (a->row > b->row) return 1;
    if (a->col < b->col) return -1;
    if (a->col > b->col) return 1;
    if (a->orientation < b->orientation) return -1;
    if (a->orientation > b->orientation) return 1;
    return 0;
}

static HPRQPStatus symmetrize_q_for_mps(HPRQPMatrixCSR *dest, const HPRQPMatrixCSR *src)
{
    HPRQPStatus status;
    size_t row;
    size_t idx;
    size_t entry_count = src->nnz;
    size_t output_nnz = 0;
    MpsSymEntry *entries;

    if (dest == nullptr || src == nullptr || !matrix_csr_is_valid(src) ||
        src->nrow != src->ncol) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    entries = static_cast<MpsSymEntry *>(
        malloc((entry_count > 0 ? entry_count : 1) * sizeof(MpsSymEntry)));
    if (entries == nullptr) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    entry_count = 0;
    for (row = 0; row < src->nrow; ++row) {
        for (idx = src->row_ptr[row]; idx < src->row_ptr[row + 1]; ++idx) {
            const size_t col = src->col_ind[idx];
            const double value = src->values[idx];
            if (col >= src->ncol || !std::isfinite(value)) {
                free(entries);
                return HPRQP_STATUS_INVALID_ARGUMENT;
            }
            entries[entry_count].row = row < col ? row : col;
            entries[entry_count].col = row < col ? col : row;
            entries[entry_count].value = value;
            entries[entry_count].orientation =
                row == col ? 0 : (row < col ? 1 : -1);
            ++entry_count;
        }
    }

    qsort(entries, entry_count, sizeof(MpsSymEntry), compare_mps_sym_entry);

    /*
     * QUADOBJ/QMATRIX files occur in both triangular and full-symmetric form.
     * Group by the unordered index pair before expanding to a symmetric CSR:
     * a one-sided record is mirrored, while two-sided records are averaged.
     * The latter is the symmetric part of a full matrix and, importantly,
     * does not double an already symmetric off-diagonal coefficient.
     */
    {
        size_t write_idx = 0;
        idx = 0;
        while (idx < entry_count) {
            const size_t first = idx;
            double upper_sum = 0.0;
            double lower_sum = 0.0;
            int have_upper = 0;
            int have_lower = 0;

            while (idx < entry_count &&
                   entries[idx].row == entries[first].row &&
                   entries[idx].col == entries[first].col) {
                if (entries[idx].orientation < 0) {
                    lower_sum += entries[idx].value;
                    have_lower = 1;
                } else {
                    upper_sum += entries[idx].value;
                    have_upper = 1;
                }
                ++idx;
            }

            entries[write_idx].row = entries[first].row;
            entries[write_idx].col = entries[first].col;
            entries[write_idx].orientation = 0;
            if (entries[first].row == entries[first].col) {
                entries[write_idx].value = upper_sum;
            } else if (have_upper && have_lower) {
                entries[write_idx].value = 0.5 * (upper_sum + lower_sum);
            } else {
                entries[write_idx].value = have_upper ? upper_sum : lower_sum;
            }
            if (!std::isfinite(entries[write_idx].value)) {
                free(entries);
                return HPRQP_STATUS_INVALID_ARGUMENT;
            }
            ++write_idx;
        }
        entry_count = write_idx;
    }

    for (idx = 0; idx < entry_count; ++idx) {
        output_nnz += entries[idx].row == entries[idx].col ? 1U : 2U;
    }

    status = matrix_csr_alloc(dest, src->nrow, src->ncol, output_nnz);
    if (status != HPRQP_STATUS_OK) {
        free(entries);
        return status;
    }

    for (idx = 0; idx < entry_count; ++idx) {
        dest->row_ptr[entries[idx].row + 1] += 1;
        if (entries[idx].row != entries[idx].col) {
            dest->row_ptr[entries[idx].col + 1] += 1;
        }
    }
    for (row = 0; row < src->nrow; ++row) {
        dest->row_ptr[row + 1] += dest->row_ptr[row];
    }

    {
        size_t *next = static_cast<size_t *>(
            malloc((src->nrow > 0 ? src->nrow : 1) * sizeof(size_t)));
        if (next == nullptr) {
            matrix_csr_free(dest);
            free(entries);
            return HPRQP_STATUS_ALLOCATION_FAILED;
        }
        for (row = 0; row < src->nrow; ++row) {
            next[row] = dest->row_ptr[row];
        }
        for (idx = 0; idx < entry_count; ++idx) {
            size_t pos = next[entries[idx].row]++;
            dest->col_ind[pos] = entries[idx].col;
            dest->values[pos] = entries[idx].value;
            if (entries[idx].row != entries[idx].col) {
                pos = next[entries[idx].col]++;
                dest->col_ind[pos] = entries[idx].row;
                dest->values[pos] = entries[idx].value;
            }
        }
        free(next);
    }

    free(entries);
    return HPRQP_STATUS_OK;
}

extern "C" HPRQPStatus build_model_from_mps(const char *mps_path, HPRQPModel *model)
{
    QPSData *qps;
    CSRMatrix *csr_Q;
    HPRQPMatrixCSR A;
    HPRQPMatrixCSR Q;
    HPRQPMatrixCSR qc_linear;
    HPRQPMatrixCSR qc_diagonal;
    double *qc_signs = nullptr;
    HPRQPStatus status;
    int qc_count = 0;
    bool direct_a_path = false;
    const bool profile_setup = std::getenv("HPRQP_PROFILE_SETUP") != nullptr;
    const auto profile_start =
        profile_setup ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{};
    std::chrono::steady_clock::time_point read_done;
    std::chrono::steady_clock::time_point csr_a_done;
    std::chrono::steady_clock::time_point csr_q_done;
    std::chrono::steady_clock::time_point copies_done;
    std::chrono::steady_clock::time_point mps_sym_done;
    std::chrono::steady_clock::time_point build_done;

    if (mps_path == nullptr || model == nullptr) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model_init(model);
    A = {};
    Q = {};
    qc_linear = {};
    qc_diagonal = {};

    qps = readqps(mps_path, MPS_FREE);
    if (profile_setup) read_done = std::chrono::steady_clock::now();
    if (qps == nullptr) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    qc_count = qps->nqc;

    if (qps->nqc > 0) {
        status = build_qc_canonical_signs(&qc_signs, qps);
        if (status == HPRQP_STATUS_OK) {
            status = build_qcqp_constraint_matrices(&A, &qc_linear, qps,
                                                    qc_signs);
        }
        if (status == HPRQP_STATUS_OK) {
            status = build_qc_diagonal(&qc_diagonal, qps, qc_signs);
        }
        direct_a_path = status == HPRQP_STATUS_OK;
    } else {
        status = build_constraint_matrix_direct(&A, qps, &direct_a_path);
    }
    if (profile_setup) csr_a_done = std::chrono::steady_clock::now();
    csr_Q = coo_to_csr(qps->nvar, qps->nvar, qps->qnnz, qps->qrows, qps->qcols, qps->qvals);
    if (profile_setup) csr_q_done = std::chrono::steady_clock::now();
    if (status != HPRQP_STATUS_OK || csr_Q == nullptr) {
        matrix_csr_free(&A);
        matrix_csr_free(&qc_linear);
        matrix_csr_free(&qc_diagonal);
        if (csr_Q != nullptr) {
            csr_free(csr_Q);
        }
        free(qc_signs);
        qpsdata_free(qps);
        return status != HPRQP_STATUS_OK ? status : HPRQP_STATUS_ALLOCATION_FAILED;
    }

    status = copy_csr_matrix(&Q, csr_Q);
    if (status != HPRQP_STATUS_OK) {
        matrix_csr_free(&A);
        matrix_csr_free(&qc_linear);
        matrix_csr_free(&qc_diagonal);
        csr_free(csr_Q);
        free(qc_signs);
        qpsdata_free(qps);
        return status;
    }
    if (profile_setup) copies_done = std::chrono::steady_clock::now();

    {
        HPRQPMatrixCSR sym_Q = {};
        status = symmetrize_q_for_mps(&sym_Q, &Q);
        if (status != HPRQP_STATUS_OK) {
            matrix_csr_free(&A);
            matrix_csr_free(&Q);
            matrix_csr_free(&qc_linear);
            matrix_csr_free(&qc_diagonal);
            csr_free(csr_Q);
            free(qc_signs);
            qpsdata_free(qps);
            return status;
        }
        matrix_csr_free(&Q);
        Q = sym_Q;
    }
    if (profile_setup) mps_sym_done = std::chrono::steady_clock::now();

    status = qps->nqc > 0
                 ? adopt_mps_qcqp_model(&Q, &A, &qc_linear, &qc_diagonal,
                                        qps, qc_signs, model)
                 : adopt_mps_model(&Q, &A, qps, model);
    if (profile_setup) build_done = std::chrono::steady_clock::now();

    matrix_csr_free(&A);
    matrix_csr_free(&Q);
    matrix_csr_free(&qc_linear);
    matrix_csr_free(&qc_diagonal);
    csr_free(csr_Q);
    free(qc_signs);
    qpsdata_free(qps);
    if (profile_setup) {
        const auto cleanup_done = std::chrono::steady_clock::now();
        const auto seconds = [](const auto &start, const auto &stop) {
            return std::chrono::duration<double>(stop - start).count();
        };
        std::fprintf(stderr,
                     "HPRQP_PROFILE mps: parse=%.6f s, build_csr_A=%.6f s, "
                     "A_path=%s, quadratic_constraints=%d, "
                     "coo_to_csr_Q=%.6f s, Q_copy=%.6f s, "
                     "first_Q_sym=%.6f s, model_adopt_preprocess=%.6f s, "
                     "cleanup=%.6f s, total=%.6f s\n",
                     seconds(profile_start, read_done),
                     seconds(read_done, csr_a_done),
                     qc_count > 0
                         ? (direct_a_path ? "direct_qc_split" : "qc_split_failed")
                         : (direct_a_path ? "direct_ordered" : "general_fallback"),
                     qc_count,
                     seconds(csr_a_done, csr_q_done),
                     seconds(csr_q_done, copies_done),
                     seconds(copies_done, mps_sym_done),
                     seconds(mps_sym_done, build_done),
                     seconds(build_done, cleanup_done),
                     seconds(profile_start, cleanup_done));
    }
    return status;
}

extern "C" HPRQPStatus build_from_mps(const char *mps_path, HPRQPModel *model)
{
    return build_model_from_mps(mps_path, model);
}
