#include "hprqp.h"

#include <math.h>
#include <stdio.h>

static int nearly_equal(double a, double b)
{
    return fabs(a - b) < 1e-12;
}

static int test_unordered_column_fallback(void)
{
    HPRQPModel model;
    HPRQPStatus status;
    const char *mps_path = HPRQP_SOURCE_DIR "/tests/data/unordered_columns_qp.mps";

    model_init(&model);
    status = build_from_mps(mps_path, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "unordered-column build_from_mps failed with status %d\n", status);
        return 1;
    }

    if (model.A.nrow != 2 || model.A.ncol != 2 || model.A.nnz != 3 ||
        model.A.row_ptr[0] != 0 || model.A.row_ptr[1] != 1 || model.A.row_ptr[2] != 3 ||
        model.A.col_ind[0] != 0 || model.A.col_ind[1] != 0 || model.A.col_ind[2] != 1 ||
        !nearly_equal(model.A.values[0], 5.0) ||
        !nearly_equal(model.A.values[1], 3.0) ||
        !nearly_equal(model.A.values[2], 2.0)) {
        fprintf(stderr, "unordered-column fallback produced the wrong A matrix\n");
        model_free(&model);
        return 1;
    }

    if (model.AT.nrow != 2 || model.AT.ncol != 2 || model.AT.nnz != 3 ||
        model.AT.row_ptr[0] != 0 || model.AT.row_ptr[1] != 2 || model.AT.row_ptr[2] != 3 ||
        model.AT.col_ind[0] != 0 || model.AT.col_ind[1] != 1 || model.AT.col_ind[2] != 1 ||
        !nearly_equal(model.AT.values[0], 5.0) ||
        !nearly_equal(model.AT.values[1], 3.0) ||
        !nearly_equal(model.AT.values[2], 2.0)) {
        fprintf(stderr, "unordered-column fallback produced the wrong AT matrix\n");
        model_free(&model);
        return 1;
    }

    model_free(&model);
    return 0;
}

static int test_simple_qcqp(void)
{
    HPRQPModel model;
    HPRQPModel copied;
    HPRQPStatus status;
    const char *mps_path = HPRQP_SOURCE_DIR "/tests/data/simple_qcqp.mps";

    model_init(&model);
    model_init(&copied);
    status = build_from_mps(mps_path, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "QCQP build_from_mps failed with status %d\n", status);
        return 1;
    }

    if (model.A.nrow != 1 || model.A.ncol != 2 || model.A.nnz != 2 ||
        !nearly_equal(model.A.values[0], 1.0) ||
        !nearly_equal(model.A.values[1], -1.0) ||
        !isinf(model.AL[0]) || model.AL[0] >= 0.0 ||
        !nearly_equal(model.AU[0], 3.0)) {
        fprintf(stderr, "ordinary rows were not separated from QC rows\n");
        model_free(&model);
        return 1;
    }
    if (model.Q.nnz != 2 || !nearly_equal(model.Q.values[0], 2.0) ||
        !nearly_equal(model.Q.values[1], 4.0)) {
        fprintf(stderr, "QCMATRIX entries contaminated the objective Q\n");
        model_free(&model);
        return 1;
    }
    if (model.qc.count != 2 || model.qc.cone_dim != 6 ||
        model.qc.linear.nrow != 2 || model.qc.linear.ncol != 2 ||
        model.qc.linear.nnz != 4 ||
        !nearly_equal(model.qc.linear.values[0], 2.0) ||
        !nearly_equal(model.qc.linear.values[1], -1.0) ||
        !nearly_equal(model.qc.linear.values[2], 1.0) ||
        !nearly_equal(model.qc.linear.values[3], 3.0) ||
        model.qc.diagonal.nnz != 2 ||
        model.qc.diagonal.col_ind[0] != 0 ||
        model.qc.diagonal.col_ind[1] != 1 ||
        /* QCMATRIX diagonal coefficients are polynomial coefficients;
         * canonical 0.5*x'Q_i*x storage therefore doubles them. */
        !nearly_equal(model.qc.diagonal.values[0], 4.0) ||
        !nearly_equal(model.qc.diagonal.values[1], 8.0) ||
        !nearly_equal(model.qc.offset[0], -4.0) ||
        !nearly_equal(model.qc.offset[1], -5.0) ||
        model.qc.cone_ptr[0] != 0 || model.qc.cone_ptr[1] != 3 ||
        model.qc.cone_ptr[2] != 6 ||
        !model.qc.disjoint_quadratic_support) {
        fprintf(stderr, "QCQP single-copy model is wrong\n");
        model_free(&model);
        return 1;
    }

    status = model_copy(&copied, &model);
    if (status != HPRQP_STATUS_OK || copied.qc.count != 2 ||
        copied.qc.linear.values == model.qc.linear.values ||
        copied.qc.diagonal.values == model.qc.diagonal.values ||
        copied.qc.offset == model.qc.offset ||
        copied.qc.cone_ptr == model.qc.cone_ptr) {
        fprintf(stderr, "QCQP model_copy did not make an independent copy\n");
        model_free(&copied);
        model_free(&model);
        return 1;
    }

    model_free(&copied);
    model_free(&model);
    return 0;
}

static int test_full_objective_qmatrix(const char *relative_path,
                                       double expected_off_diagonal)
{
    HPRQPModel model;
    HPRQPStatus status;
    char mps_path[1024];

    if (snprintf(mps_path, sizeof(mps_path), "%s/tests/data/%s",
                 HPRQP_SOURCE_DIR, relative_path) >= (int)sizeof(mps_path)) {
        fprintf(stderr, "test MPS path is too long\n");
        return 1;
    }

    model_init(&model);
    status = build_from_mps(mps_path, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "%s build_from_mps failed with status %d\n",
                relative_path, status);
        return 1;
    }

    if (model.Q.nrow != 2 || model.Q.ncol != 2 || model.Q.nnz != 4 ||
        model.Q.row_ptr[0] != 0 || model.Q.row_ptr[1] != 2 ||
        model.Q.row_ptr[2] != 4 ||
        model.Q.col_ind[0] != 0 || model.Q.col_ind[1] != 1 ||
        model.Q.col_ind[2] != 0 || model.Q.col_ind[3] != 1 ||
        !nearly_equal(model.Q.values[0], 2.0) ||
        !nearly_equal(model.Q.values[1], expected_off_diagonal) ||
        !nearly_equal(model.Q.values[2], expected_off_diagonal) ||
        !nearly_equal(model.Q.values[3], 4.0)) {
        fprintf(stderr, "%s objective Q was symmetrized incorrectly\n",
                relative_path);
        model_free(&model);
        return 1;
    }

    model_free(&model);
    return 0;
}

static int test_lower_negative_qcqp(void)
{
    HPRQPModel model;
    HPRQPStatus status;
    const char *mps_path =
        HPRQP_SOURCE_DIR "/tests/data/lower_negative_qcqp.mps";

    model_init(&model);
    status = build_from_mps(mps_path, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "lower-negative QCQP build failed with status %d\n",
                status);
        return 1;
    }

    if (model.A.nrow != 0 || model.qc.count != 1 ||
        model.qc.linear.nrow != 1 || model.qc.linear.nnz != 1 ||
        model.qc.linear.col_ind[0] != 0 ||
        !nearly_equal(model.qc.linear.values[0], 2.0) ||
        model.qc.diagonal.nrow != 1 || model.qc.diagonal.nnz != 1 ||
        model.qc.diagonal.col_ind[0] != 0 ||
        !nearly_equal(model.qc.diagonal.values[0], 2.0) ||
        !nearly_equal(model.qc.offset[0], -4.0) ||
        model.qc.cone_ptr[0] != 0 || model.qc.cone_ptr[1] != 3) {
        fprintf(stderr, "lower-negative QC row was not canonicalized\n");
        model_free(&model);
        return 1;
    }

    model_free(&model);
    return 0;
}

int main(void)
{
    HPRQPModel model;
    HPRQPStatus status;
    const char *mps_path = HPRQP_SOURCE_DIR "/tests/data/simple_qp.mps";

    model_init(&model);
    status = build_from_mps(mps_path, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "build_from_mps failed with status %d\n", status);
        return 1;
    }

    if (model.A.nrow != 2 || model.A.ncol != 2 || model.A.nnz != 4) {
        fprintf(stderr, "A dimensions are wrong\n");
        model_free(&model);
        return 1;
    }

    if (model.Q.nrow != 2 || model.Q.ncol != 2 || model.Q.nnz != 4) {
        fprintf(stderr, "Q dimensions are wrong\n");
        model_free(&model);
        return 1;
    }

    if (!nearly_equal(model.Q.values[0], 2.0) || !nearly_equal(model.Q.values[1], 1.0) ||
        !nearly_equal(model.Q.values[2], 1.0) || !nearly_equal(model.Q.values[3], 4.0)) {
        fprintf(stderr, "Q values are wrong after symmetrization\n");
        model_free(&model);
        return 1;
    }

    if (model.A.nrow != 2) {
        fprintf(stderr, "unexpected row deletion in mps reader test\n");
        model_free(&model);
        return 1;
    }

    if (!nearly_equal(model.c[0], 1.0) || !nearly_equal(model.c[1], 2.0)) {
        fprintf(stderr, "objective coefficients are wrong\n");
        model_free(&model);
        return 1;
    }

    if (!nearly_equal(model.AL[0], 1.0) || !isinf(model.AL[1]) || model.AL[1] >= 0.0) {
        fprintf(stderr, "constraint lower bounds are wrong\n");
        model_free(&model);
        return 1;
    }

    if (!isinf(model.AU[0]) || model.AU[0] <= 0.0 || !nearly_equal(model.AU[1], 2.0)) {
        fprintf(stderr, "constraint upper bounds are wrong\n");
        model_free(&model);
        return 1;
    }

    if (!nearly_equal(model.l[0], 0.0) || !nearly_equal(model.u[0], 10.0) ||
        !nearly_equal(model.l[1], -5.0) || !nearly_equal(model.u[1], 5.0)) {
        fprintf(stderr, "variable bounds are wrong\n");
        model_free(&model);
        return 1;
    }

    if (model.AT.nrow != 2 || model.AT.ncol != 2 || model.AT.nnz != 4) {
        fprintf(stderr, "AT dimensions are wrong\n");
        model_free(&model);
        return 1;
    }

    model_free(&model);
    if (test_unordered_column_fallback() != 0) {
        return 1;
    }
    if (test_simple_qcqp() != 0) {
        return 1;
    }
    if (test_full_objective_qmatrix("full_symmetric_qp.mps", 3.0) != 0) {
        return 1;
    }
    if (test_full_objective_qmatrix("asymmetric_full_qp.mps", 3.0) != 0) {
        return 1;
    }
    if (test_lower_negative_qcqp() != 0) {
        return 1;
    }
    printf("test_mps_reader passed\n");
    return 0;
}
