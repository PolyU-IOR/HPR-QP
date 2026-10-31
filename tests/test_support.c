#include "test_support.h"

#include <string.h>

typedef struct HPRQPTestMatrixSpec {
    size_t nrow;
    size_t ncol;
    size_t nnz;
    const size_t *row_ptr;
    const size_t *col_ind;
    const double *values;
} HPRQPTestMatrixSpec;

static HPRQPStatus test_build_matrix_from_spec(HPRQPMatrixCSR *matrix,
                                               const HPRQPTestMatrixSpec *spec)
{
    HPRQPStatus status;

    status = matrix_csr_alloc(matrix, spec->nrow, spec->ncol, spec->nnz);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    memcpy(matrix->row_ptr, spec->row_ptr, (spec->nrow + 1) * sizeof(size_t));
    if (spec->nnz > 0) {
        memcpy(matrix->col_ind, spec->col_ind, spec->nnz * sizeof(size_t));
        memcpy(matrix->values, spec->values, spec->nnz * sizeof(double));
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus test_build_model_from_specs(const HPRQPTestMatrixSpec *q_spec,
                                               const HPRQPTestMatrixSpec *a_spec,
                                               const double *c,
                                               const double *AL,
                                               const double *AU,
                                               const double *l,
                                               const double *u,
                                               HPRQPMatrixCSR *Q,
                                               HPRQPMatrixCSR *A,
                                               HPRQPModel *model)
{
    HPRQPStatus status;

    status = test_build_matrix_from_spec(Q, q_spec);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = test_build_matrix_from_spec(A, a_spec);
    if (status != HPRQP_STATUS_OK) {
        matrix_csr_free(Q);
        return status;
    }

    status = build_model_from_arrays(Q, A, c, AL, AU, l, u, 0.0, model);
    if (status != HPRQP_STATUS_OK) {
        matrix_csr_free(A);
        matrix_csr_free(Q);
        return status;
    }

    return HPRQP_STATUS_OK;
}

void test_params_set_no_scaling(HPRQPParams *params)
{
    params_set_defaults(params);
    params->use_scaling = 0;
}

HPRQPStatus test_build_model_case(HPRQPTestModelCase case_id,
                                  HPRQPMatrixCSR *Q,
                                  HPRQPMatrixCSR *A,
                                  HPRQPModel *model)
{
    switch (case_id) {
    case HPRQP_TEST_CASE_STANDARD: {
        static const size_t q_row_ptr[] = {0, 1, 2};
        static const size_t q_col_ind[] = {0, 1};
        static const double q_values[] = {2.0, 4.0};
        static const size_t a_row_ptr[] = {0, 2};
        static const size_t a_col_ind[] = {0, 1};
        static const double a_values[] = {1.0, 1.0};
        static const double c[] = {2.0, 1.0};
        static const double AL[] = {2.0};
        static const double AU[] = {2.0};
        static const double l[] = {0.0, 0.0};
        static const double u[] = {10.0, 10.0};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 2, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {1, 2, 2, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_STANDARD_NOC: {
        static const size_t q_row_ptr[] = {0, 1, 2};
        static const size_t q_col_ind[] = {0, 1};
        static const double q_values[] = {2.0, 4.0};
        static const size_t a_row_ptr[] = {0, 2};
        static const size_t a_col_ind[] = {0, 1};
        static const double a_values[] = {1.0, 1.0};
        static const double c[] = {2.0, 1.0};
        static const double AL[] = {2.0};
        static const double AU[] = {2.0};
        static const double l[] = {-INFINITY, -INFINITY};
        static const double u[] = {INFINITY, INFINITY};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 2, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {1, 2, 2, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_SCALE: {
        static const size_t q_row_ptr[] = {0, 1, 2};
        static const size_t q_col_ind[] = {0, 1};
        static const double q_values[] = {4.0, 9.0};
        static const size_t a_row_ptr[] = {0, 2};
        static const size_t a_col_ind[] = {0, 1};
        static const double a_values[] = {16.0, 25.0};
        static const double c[] = {2.0, 4.0};
        static const double AL[] = {2.0};
        static const double AU[] = {6.0};
        static const double l[] = {0.0, -1.0};
        static const double u[] = {10.0, 3.0};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 2, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {1, 2, 2, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_RESIDUALS: {
        static const size_t q_row_ptr[] = {0, 1, 2};
        static const size_t q_col_ind[] = {0, 1};
        static const double q_values[] = {2.0, 4.0};
        static const size_t a_row_ptr[] = {0, 2};
        static const size_t a_col_ind[] = {0, 1};
        static const double a_values[] = {1.0, 2.0};
        static const double c[] = {1.0, -1.0};
        static const double AL[] = {0.0};
        static const double AU[] = {100.0};
        static const double l[] = {0.0, 0.0};
        static const double u[] = {10.0, 10.0};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 2, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {1, 2, 2, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_DENSE_Q: {
        static const size_t q_row_ptr[] = {0, 2, 4};
        static const size_t q_col_ind[] = {0, 1, 0, 1};
        static const double q_values[] = {2.0, 1.0, 1.0, 2.0};
        static const size_t a_row_ptr[] = {0, 2};
        static const size_t a_col_ind[] = {0, 1};
        static const double a_values[] = {1.0, 2.0};
        static const double c[] = {0.0, 0.0};
        static const double AL[] = {-1.0};
        static const double AU[] = {1.0};
        static const double l[] = {-10.0, -10.0};
        static const double u[] = {10.0, 10.0};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 4, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {1, 2, 2, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_CONTROL_NONDIAG: {
        static const size_t q_row_ptr[] = {0, 2, 4};
        static const size_t q_col_ind[] = {0, 1, 0, 1};
        static const double q_values[] = {2.0, 0.5, 0.5, 3.0};
        static const size_t a_row_ptr[] = {0, 2};
        static const size_t a_col_ind[] = {0, 1};
        static const double a_values[] = {1.0, -1.0};
        static const double c[] = {1.0, -0.5};
        static const double AL[] = {0.0};
        static const double AU[] = {1.5};
        static const double l[] = {-10.0, -10.0};
        static const double u[] = {10.0, 10.0};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 4, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {1, 2, 2, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_EMPTY_A: {
        static const size_t q_row_ptr[] = {0, 1, 2};
        static const size_t q_col_ind[] = {0, 1};
        static const double q_values[] = {2.0, 3.0};
        static const size_t a_row_ptr[] = {0};
        static const double c[] = {1.0, -0.5};
        static const double l[] = {-INFINITY, -INFINITY};
        static const double u[] = {INFINITY, INFINITY};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 2, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {0, 2, 0, a_row_ptr, NULL, NULL};

        return test_build_model_from_specs(&q_spec, &a_spec, c, NULL, NULL, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_TALL_DIAG: {
        static const size_t q_row_ptr[] = {0, 1, 2};
        static const size_t q_col_ind[] = {0, 1};
        static const double q_values[] = {3.0, 5.0};
        static const size_t a_row_ptr[] = {0, 2, 4, 6};
        static const size_t a_col_ind[] = {0, 1, 0, 1, 0, 1};
        static const double a_values[] = {1.0, 1.0, 1.0, -1.0, 2.0, 0.5};
        static const double c[] = {1.5, -0.5};
        static const double AL[] = {1.0, -0.5, 0.25};
        static const double AU[] = {1.0, -0.5, 0.25};
        static const double l[] = {0.0, -1.0};
        static const double u[] = {2.0, 3.0};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 2, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {3, 2, 6, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_DISPATCH_1D: {
        static const size_t q_row_ptr[] = {0, 1};
        static const size_t q_col_ind[] = {0};
        static const double q_values[] = {1.0};
        static const size_t a_row_ptr[] = {0};
        static const double c[] = {0.0};
        static const double l[] = {-1.0};
        static const double u[] = {1.0};
        static const HPRQPTestMatrixSpec q_spec = {1, 1, 1, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {0, 1, 0, a_row_ptr, NULL, NULL};

        return test_build_model_from_specs(&q_spec, &a_spec, c, NULL, NULL, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_GPU_SPMV_1D: {
        static const size_t q_row_ptr[] = {0, 1};
        static const size_t q_col_ind[] = {0};
        static const double q_values[] = {2.0};
        static const size_t a_row_ptr[] = {0, 1};
        static const size_t a_col_ind[] = {0};
        static const double a_values[] = {3.0};
        static const double c[] = {0.0};
        static const double AL[] = {-1.0};
        static const double AU[] = {1.0};
        static const double l[] = {-10.0};
        static const double u[] = {10.0};
        static const HPRQPTestMatrixSpec q_spec = {1, 1, 1, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {1, 1, 1, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_GPU_ITERATION_STANDARD: {
        static const size_t q_row_ptr[] = {0, 2, 4};
        static const size_t q_col_ind[] = {0, 1, 0, 1};
        static const double q_values[] = {2.0, 0.5, 0.5, 3.0};
        static const size_t a_row_ptr[] = {0, 2};
        static const size_t a_col_ind[] = {0, 1};
        static const double a_values[] = {1.0, -2.0};
        static const double c[] = {0.5, -0.25};
        static const double AL[] = {-0.5};
        static const double AU[] = {0.75};
        static const double l[] = {-1.0, -1.0};
        static const double u[] = {1.0, 1.0};
        static const HPRQPTestMatrixSpec q_spec = {2, 2, 4, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {1, 2, 2, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_ZERO_1D: {
        static const size_t q_row_ptr[] = {0, 0};
        static const size_t a_row_ptr[] = {0};
        static const double c[] = {0.0};
        static const double l[] = {-1.0};
        static const double u[] = {1.0};
        static const HPRQPTestMatrixSpec q_spec = {1, 1, 0, q_row_ptr, NULL, NULL};
        static const HPRQPTestMatrixSpec a_spec = {0, 1, 0, a_row_ptr, NULL, NULL};

        return test_build_model_from_specs(&q_spec, &a_spec, c, NULL, NULL, l, u, Q, A, model);
    }
    case HPRQP_TEST_CASE_PARALLEL_SINGULAR: {
        static const size_t q_row_ptr[] = {0, 1, 1, 2};
        static const size_t q_col_ind[] = {0, 2};
        static const double q_values[] = {2.0, 1.0};
        static const size_t a_row_ptr[] = {0, 2, 4};
        static const size_t a_col_ind[] = {0, 1, 1, 2};
        static const double a_values[] = {1.0, 1.0, 1.0, 1.0};
        static const double c[] = {-2.0, -0.5, -1.0};
        static const double AL[] = {0.25, 0.10};
        static const double AU[] = {1.50, 1.25};
        static const double l[] = {0.0, 0.0, 0.0};
        static const double u[] = {1.0, 1.0, 1.0};
        static const HPRQPTestMatrixSpec q_spec = {3, 3, 2, q_row_ptr, q_col_ind, q_values};
        static const HPRQPTestMatrixSpec a_spec = {2, 3, 4, a_row_ptr, a_col_ind, a_values};

        return test_build_model_from_specs(&q_spec, &a_spec, c, AL, AU, l, u, Q, A, model);
    }
    }

    return HPRQP_STATUS_INVALID_ARGUMENT;
}
