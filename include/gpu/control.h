#ifndef HPRQP_GPU_CONTROL_H
#define HPRQP_GPU_CONTROL_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus gpu_workspace_setup(HPRQPGPUWorkspace *workspace,
                                const HPRQPGPUProblem *model,
                                const HPRQPParams *params);
HPRQPStatus gpu_compute_m_norm_value(HPRQPGPUWorkspace *gpu_workspace,
                                     const HPRQPGPUProblem *gpu_problem,
                                     double *out_value);
HPRQPStatus gpu_sigma_update_value(HPRQPGPUWorkspace *gpu_workspace,
                                   const HPRQPGPUProblem *gpu_problem,
                                   const HPRQPParams *params,
                                   const HPRQPRestart *restart_info,
                                   const HPRQPResiduals *residuals,
                                   double *out_sigma);
HPRQPStatus gpu_restart_apply_value(HPRQPGPUWorkspace *gpu_workspace,
                                    const HPRQPGPUProblem *gpu_problem);
HPRQPStatus gpu_compute_m_norm_parallel_value(HPRQPGPUWorkspace *gpu_workspace,
                                              double *out_value);
HPRQPStatus gpu_parallel_update_metric_value(HPRQPGPUWorkspace *gpu_workspace,
                                             const HPRQPGPUProblem *gpu_problem,
                                             const HPRQPParams *params,
                                             HPRQPRestart *restart_info,
                                             const HPRQPResiduals *residuals);
HPRQPStatus gpu_restart_apply_parallel_value(HPRQPGPUWorkspace *gpu_workspace);

#ifdef __cplusplus
}
#endif

#endif
