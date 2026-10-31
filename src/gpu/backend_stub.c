#include "gpu/gpu.h"

#include "gpu/iteration.h"

int gpu_backend_available(void)
{
    return 0;
}

HPRQPStatus gpu_iteration_step(HPRQPGPUWorkspace *gpu_workspace,
                               const HPRQPGPUProblem *gpu_problem,
                               size_t inner_iter)
{
    (void)gpu_workspace;
    (void)gpu_problem;
    (void)inner_iter;
    return HPRQP_STATUS_NOT_IMPLEMENTED;
}