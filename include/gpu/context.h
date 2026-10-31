#ifndef HPRQP_GPU_CONTEXT_H
#define HPRQP_GPU_CONTEXT_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_context_init(HPRQPGPUContext *context);
void gpu_context_free(HPRQPGPUContext *context);
HPRQPStatus gpu_context_open(HPRQPGPUContext *context, int device_number);

void gpu_vector_init(HPRQPGPUVector *vector);
void gpu_vector_free(HPRQPGPUVector *vector);
HPRQPStatus gpu_vector_alloc(HPRQPGPUVector *vector, size_t length);
HPRQPStatus gpu_vector_copy_from_host(HPRQPGPUVector *vector, const double *host_values, size_t length);
HPRQPStatus gpu_vector_copy_to_host(const HPRQPGPUVector *vector, double *host_values, size_t length);

void gpu_matrix_init(HPRQPGPUMatrixCSR *matrix);
void gpu_matrix_free(HPRQPGPUMatrixCSR *matrix);
HPRQPStatus gpu_matrix_upload(HPRQPGPUMatrixCSR *gpu_matrix, const HPRQPMatrixCSR *cpu_matrix);

void gpu_problem_init(HPRQPGPUProblem *model);
void gpu_problem_free(HPRQPGPUProblem *model);
HPRQPStatus gpu_problem_upload(HPRQPGPUProblem *gpu_problem, const HPRQPModel *cpu_problem);

#ifdef __cplusplus
}
#endif

#endif