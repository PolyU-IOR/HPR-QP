#ifndef HPRQP_GPU_RESIDUALS_H
#define HPRQP_GPU_RESIDUALS_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus gpu_compute_residuals_value(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPGPUProblem *gpu_problem,
                                        HPRQPResiduals *residuals,
                                        size_t iter);

#ifdef __cplusplus
}
#endif

#endif