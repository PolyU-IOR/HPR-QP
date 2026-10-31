#ifndef HPRQP_TEST_SUPPORT_H
#define HPRQP_TEST_SUPPORT_H

#include "internal/hprqp_internal.h"

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum HPRQPTestModelCase {
    HPRQP_TEST_CASE_STANDARD = 0,
    HPRQP_TEST_CASE_STANDARD_NOC,
    HPRQP_TEST_CASE_SCALE,
    HPRQP_TEST_CASE_RESIDUALS,
    HPRQP_TEST_CASE_DENSE_Q,
    HPRQP_TEST_CASE_CONTROL_NONDIAG,
    HPRQP_TEST_CASE_EMPTY_A,
    HPRQP_TEST_CASE_TALL_DIAG,
    HPRQP_TEST_CASE_DISPATCH_1D,
    HPRQP_TEST_CASE_GPU_SPMV_1D,
    HPRQP_TEST_CASE_GPU_ITERATION_STANDARD,
    HPRQP_TEST_CASE_ZERO_1D,
    HPRQP_TEST_CASE_PARALLEL_SINGULAR,
} HPRQPTestModelCase;

HPRQPStatus test_build_model_case(HPRQPTestModelCase case_id,
                                  HPRQPMatrixCSR *Q,
                                  HPRQPMatrixCSR *A,
                                  HPRQPModel *model);

void test_params_set_no_scaling(HPRQPParams *params);

#ifdef __cplusplus
}
#endif

static inline int test_nearly_equal_abs(double a, double b, double tol)
{
    return fabs(a - b) <= tol;
}

static inline int test_nearly_equal_rel(double a, double b, double tol)
{
    double scale = fmax(1.0, fmax(fabs(a), fabs(b)));

    return fabs(a - b) <= tol * scale;
}

#define TEST_ASSERT(cond, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, __VA_ARGS__); \
            fputc('\n', stderr); \
            exit_code = 1; \
            goto cleanup; \
        } \
    } while (0)

#define TEST_ASSERT_OK(expr) \
    do { \
        HPRQPStatus test_status__ = (expr); \
        if (test_status__ != HPRQP_STATUS_OK) { \
            fprintf(stderr, "%s failed with status=%d\n", #expr, (int)test_status__); \
            exit_code = 1; \
            goto cleanup; \
        } \
    } while (0)

#endif
