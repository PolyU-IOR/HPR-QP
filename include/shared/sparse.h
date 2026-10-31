#ifndef HPRQP_SPARSE_H
#define HPRQP_SPARSE_H

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus matrix_csr_alloc(HPRQPMatrixCSR *matrix, size_t nrow, size_t ncol, size_t nnz);
void matrix_csr_free(HPRQPMatrixCSR *matrix);
HPRQPStatus matrix_csr_copy(HPRQPMatrixCSR *dest, const HPRQPMatrixCSR *src);
HPRQPStatus matrix_csr_transpose(HPRQPMatrixCSR *dest, const HPRQPMatrixCSR *src);
HPRQPStatus matrix_csr_matvec(const HPRQPMatrixCSR *matrix, const double *x, double *y);
HPRQPStatus matrix_csr_transpose_matvec(const HPRQPMatrixCSR *matrix, const double *x, double *y);
int matrix_csr_is_valid(const HPRQPMatrixCSR *matrix);

#ifdef __cplusplus
}
#endif

#endif
