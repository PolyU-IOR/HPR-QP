#ifndef HPRQP_TEST_EIGEN_RETRY_H
#define HPRQP_TEST_EIGEN_RETRY_H

#include "test_support.h"

/* A=[16 I, (1,-2,1)^T], Q=diag(0,0,0,q). The original affine probe
 * misses A Q A^T, whose nonzero eigenvalue is 6q (6q^2 for Q^2).
 * The unchanged AA^T estimate 1.05*256 already bounds 256+6; this
 * isolates the moment-probe bug rather than changing the AA^T path. */
static HPRQPStatus test_build_eigen_retry_case(double q, int zero_coupling,
                                               HPRQPModel *model)
{
    HPRQPMatrixCSR Q = {0}, A = {0};
    HPRQPStatus status;
    const double c[] = {0,0,0,0}, row_bounds[] = {0,0,0};
    const double lower[] = {-1,-1,-1,-1}, upper[] = {1,1,1,1};
    const double v[] = {1,-2,1};
    size_t i;
    status = matrix_csr_alloc(&Q,4,4,1);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    for (i=0;i<4;++i) Q.row_ptr[i]=0;
    Q.row_ptr[4]=1; Q.col_ind[0]=3; Q.values[0]=q;
    status = matrix_csr_alloc(&A,3,4,6);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    for (i=0;i<3;++i) {
        A.row_ptr[i]=2*i;
        A.col_ind[2*i]=i; A.values[2*i]=16;
        A.col_ind[2*i+1]=3; A.values[2*i+1]=zero_coupling?0:v[i];
    }
    A.row_ptr[3]=6;
    status=build_model_from_arrays(&Q,&A,c,row_bounds,row_bounds,
                                    lower,upper,0,model);
cleanup:
    matrix_csr_free(&A); matrix_csr_free(&Q);
    return status;
}

#endif
