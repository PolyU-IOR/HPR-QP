#ifndef HPRQP_GPU_SPMV_H
#define HPRQP_GPU_SPMV_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus gpu_workspace_prepare_spmv(HPRQPGPUWorkspace *workspace,
                                       const HPRQPGPUProblem *model,
                                       int verbose);
HPRQPStatus gpu_workspace_prepare_selected_spmv_graphs(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *model,
    int verbose);
HPRQPStatus gpu_workspace_select_spmv_modes(HPRQPGPUWorkspace *workspace,
                                            const HPRQPGPUProblem *model,
                                            const HPRQPParams *params);
HPRQPStatus gpu_spmv_apply_q_w(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_q_w_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_q_x_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_q_dw(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_q_atdy(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_a_tempv(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_a_x_hat(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_a_x_hat_graph(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_a_x_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_at_y(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_at_y_graph(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_at_y_bar(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_at_dy(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);
HPRQPStatus gpu_spmv_apply_at_y_bar_residual(HPRQPGPUWorkspace *workspace, const HPRQPGPUProblem *model);

#ifdef __cplusplus
}
#endif

#endif
