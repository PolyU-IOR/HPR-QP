#ifndef HPRQP_GPU_WORKSPACE_H
#define HPRQP_GPU_WORKSPACE_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void gpu_workspace_init(HPRQPGPUWorkspace *workspace);
void gpu_workspace_free(HPRQPGPUWorkspace *workspace);
HPRQPStatus gpu_workspace_allocate(HPRQPGPUWorkspace *workspace,
                                   const HPRQPGPUProblem *model,
                                   const HPRQPParams *params);
HPRQPStatus gpu_workspace_upload_scaling(HPRQPGPUWorkspace *workspace,
                                         const HPRQPScaling *scaling);
HPRQPStatus gpu_workspace_sync_from_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPWorkspace *cpu_workspace);
HPRQPStatus gpu_workspace_copy_main_state_to_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                                 HPRQPWorkspace *cpu_workspace);
HPRQPStatus gpu_workspace_copy_history_state_to_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                                    HPRQPWorkspace *cpu_workspace);
HPRQPStatus gpu_workspace_copy_check_state_to_cpu(HPRQPGPUWorkspace *gpu_workspace,
                                                  HPRQPWorkspace *cpu_workspace);
HPRQPStatus gpu_workspace_copy_output_unscaled_to_host(HPRQPGPUWorkspace *gpu_workspace,
                                                       double *x,
                                                       double *w,
                                                       double *z,
                                                       double *y);

#ifdef __cplusplus
}
#endif

#endif