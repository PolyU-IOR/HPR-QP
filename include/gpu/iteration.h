#ifndef HPRQP_GPU_ITERATION_H
#define HPRQP_GPU_ITERATION_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus gpu_iteration_step(HPRQPGPUWorkspace *gpu_workspace,
                               const HPRQPGPUProblem *gpu_problem,
                               size_t inner_iter);
HPRQPStatus gpu_iteration_step_parallel(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPGPUProblem *gpu_problem,
                                        size_t inner_iter);
HPRQPStatus gpu_iteration_batch_parallel(HPRQPGPUWorkspace *gpu_workspace,
                                         const HPRQPGPUProblem *gpu_problem,
                                         size_t inner_iter,
                                         size_t batch_count);

#ifdef __cplusplus
}
#endif

#endif
