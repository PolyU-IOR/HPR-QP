#include "hprqp.h"

#include "shared/vector.h"

#include <math.h>
#include <stdio.h>

static int nearly_equal(double a, double b)
{
    return fabs(a - b) < 1e-12;
}

int main(void)
{
    HPRQPMatrixCSR matrix;
    HPRQPMatrixCSR q_matrix;
    HPRQPModel model;
    double x[3] = {1.0, 2.0, 3.0};
    double y[2] = {0.0, 0.0};
    double yt[3] = {0.0, 0.0, 0.0};
    double projected[3] = {0.0, 0.0, 0.0};
    double lower[3] = {0.0, 1.5, 2.5};
    double upper[3] = {1.0, 2.5, 2.8};
    double c[3] = {1.0, -1.0, 0.5};
    double al[2] = {-1.0, 0.0};
    double au[2] = {2.0, 3.0};
    double ql[3] = {-2.0, -2.0, -2.0};
    double qu[3] = {2.0, 2.0, 2.0};
    HPRQPStatus status;

    model_init(&model);

    status = matrix_csr_alloc(&matrix, 2, 3, 4);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    matrix.row_ptr[0] = 0;
    matrix.row_ptr[1] = 2;
    matrix.row_ptr[2] = 4;

    matrix.col_ind[0] = 0;
    matrix.col_ind[1] = 2;
    matrix.col_ind[2] = 1;
    matrix.col_ind[3] = 2;

    matrix.values[0] = 1.0;
    matrix.values[1] = 4.0;
    matrix.values[2] = 2.0;
    matrix.values[3] = 5.0;

    status = matrix_csr_matvec(&matrix, x, y);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "matvec failed\n");
        matrix_csr_free(&matrix);
        return 1;
    }

    if (!nearly_equal(y[0], 13.0) || !nearly_equal(y[1], 19.0)) {
        fprintf(stderr, "matvec produced wrong result: [%f, %f]\n", y[0], y[1]);
        matrix_csr_free(&matrix);
        return 1;
    }

    status = matrix_csr_transpose_matvec(&matrix, y, yt);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "transpose matvec failed\n");
        matrix_csr_free(&matrix);
        return 1;
    }

    if (!nearly_equal(yt[0], 13.0) || !nearly_equal(yt[1], 38.0) || !nearly_equal(yt[2], 147.0)) {
        fprintf(stderr, "transpose matvec produced wrong result: [%f, %f, %f]\n", yt[0], yt[1], yt[2]);
        matrix_csr_free(&matrix);
        return 1;
    }

    vec_project_box(projected, x, lower, upper, 3);
    if (!nearly_equal(projected[0], 1.0) || !nearly_equal(projected[1], 2.0) || !nearly_equal(projected[2], 2.8)) {
        fprintf(stderr, "projection produced wrong result: [%f, %f, %f]\n", projected[0], projected[1], projected[2]);
        matrix_csr_free(&matrix);
        return 1;
    }

    status = matrix_csr_alloc(&q_matrix, 3, 3, 3);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "Q allocation failed\n");
        matrix_csr_free(&matrix);
        return 1;
    }

    q_matrix.row_ptr[0] = 0;
    q_matrix.row_ptr[1] = 1;
    q_matrix.row_ptr[2] = 2;
    q_matrix.row_ptr[3] = 3;
    q_matrix.col_ind[0] = 0;
    q_matrix.col_ind[1] = 1;
    q_matrix.col_ind[2] = 2;
    q_matrix.values[0] = 2.0;
    q_matrix.values[1] = 3.0;
    q_matrix.values[2] = 4.0;

    status = build_from_QAbc(&q_matrix, &matrix, c, al, au, ql, qu, 1.5, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "build_from_QAbc failed with status %d\n", status);
        matrix_csr_free(&q_matrix);
        matrix_csr_free(&matrix);
        return 1;
    }

    if (model.AT.nrow != 3 || model.AT.ncol != 2 || model.AT.nnz != 4) {
        fprintf(stderr, "AT dimensions are wrong\n");
        model_free(&model);
        matrix_csr_free(&q_matrix);
        matrix_csr_free(&matrix);
        return 1;
    }

    if (!nearly_equal(model.obj_constant, 1.5) || !nearly_equal(model.c[1], -1.0) || !nearly_equal(model.AL[1], 0.0)) {
        fprintf(stderr, "model data copy is wrong\n");
        model_free(&model);
        matrix_csr_free(&q_matrix);
        matrix_csr_free(&matrix);
        return 1;
    }

    model_free(&model);
    matrix_csr_free(&q_matrix);
    matrix_csr_free(&matrix);
    printf("test_sparse_helpers passed\n");
    return 0;
}
