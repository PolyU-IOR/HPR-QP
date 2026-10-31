#ifndef HPRQP_GPU_EIGS_H
#define HPRQP_GPU_EIGS_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus gpu_estimate_eigenvalues(HPRQPGPUWorkspace *workspace,
                                     const HPRQPGPUProblem *model,
                                     const HPRQPParams *params);

#ifdef __cplusplus
}
#endif

#endif
