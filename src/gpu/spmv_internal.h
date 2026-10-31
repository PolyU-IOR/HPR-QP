#ifndef HPRQP_GPU_SPMV_INTERNAL_H
#define HPRQP_GPU_SPMV_INTERNAL_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_spmv_a_init(HPRQPGPUSpMVA *spmv);
void gpu_spmv_at_init(HPRQPGPUSpMVAT *spmv);
void gpu_spmv_q_init(HPRQPGPUSpMVQ *spmv);
void gpu_spmv_a_free(HPRQPGPUSpMVA *spmv);
void gpu_spmv_at_free(HPRQPGPUSpMVAT *spmv);
void gpu_spmv_q_free(HPRQPGPUSpMVQ *spmv);

#ifdef __cplusplus
}
#endif

#endif