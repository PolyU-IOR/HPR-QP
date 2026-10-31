#include "internal/hprqp_internal.h"

#include <math.h>
#include <stdio.h>

static int nearly_equal(double a, double b)
{
    return fabs(a - b) < 1e-12;
}

int main(void)
{
    HPRQPMatrixCSR Q;
    HPRQPMatrixCSR A;
    HPRQPModel model;
    HPRQPStatus status;
    double c[2] = {1.0, 2.0};
    double AL[3] = {0.0, -1.0, -INFINITY};
    double AU[3] = {3.0, 1.0, INFINITY};
    double l[2] = {-2.0, -2.0};
    double u[2] = {2.0, 2.0};

    model_init(&model);
    status = matrix_csr_alloc(&Q, 2, 2, 3);
    if (status != HPRQP_STATUS_OK) {
        return 1;
    }
    Q.row_ptr[0] = 0;
    Q.row_ptr[1] = 2;
    Q.row_ptr[2] = 3;
    Q.col_ind[0] = 0;
    Q.col_ind[1] = 1;
    Q.col_ind[2] = 1;
    Q.values[0] = 2.0;
    Q.values[1] = 6.0;
    Q.values[2] = 4.0;

    status = matrix_csr_alloc(&A, 3, 2, 2);
    if (status != HPRQP_STATUS_OK) {
        matrix_csr_free(&Q);
        return 1;
    }
    A.row_ptr[0] = 0;
    A.row_ptr[1] = 2;
    A.row_ptr[2] = 2;
    A.row_ptr[3] = 2;
    A.col_ind[0] = 0;
    A.col_ind[1] = 1;
    A.values[0] = 1.0;
    A.values[1] = -1.0;

    status = build_model_from_arrays(&Q, &A, c, AL, AU, l, u, 0.0, &model);
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "build_model_from_arrays failed\n");
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (model.A.nrow != 1 || model.AT.ncol != 1) {
        fprintf(stderr, "row deletion failed\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (model.Q.nnz != 4) {
        fprintf(stderr, "Q symmetrization nnz is wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(model.Q.values[1], 3.0) || !nearly_equal(model.Q.values[2], 3.0)) {
        fprintf(stderr, "Q symmetrization values are wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    if (!nearly_equal(model.AL[0], 0.0) || !nearly_equal(model.AU[0], 3.0)) {
        fprintf(stderr, "bounds after row deletion are wrong\n");
        model_free(&model);
        matrix_csr_free(&Q);
        matrix_csr_free(&A);
        return 1;
    }

    model_free(&model);
    matrix_csr_free(&Q);
    matrix_csr_free(&A);
    printf("test_builder_preprocess passed\n");
    return 0;
}
