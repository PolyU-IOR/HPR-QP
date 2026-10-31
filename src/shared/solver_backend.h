#ifndef HPRQP_SOLVER_BACKEND_H
#define HPRQP_SOLVER_BACKEND_H

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HPRQPSolverBackendContext {
    HPRQPWorkspace *workspace;
    const HPRQPModel *model;
    const HPRQPScaling *scaling;
    const HPRQPParams *params;
    void *data;
} HPRQPSolverBackendContext;

typedef struct HPRQPSolverBackendOps {
    HPRQPStatus (*compute_residuals)(HPRQPSolverBackendContext *context,
                                     HPRQPResiduals *residuals,
                                     size_t iter);
    HPRQPStatus (*copy_output_iterates)(HPRQPSolverBackendContext *context,
                                        double *x_bar,
                                        double *w_bar,
                                        double *z_bar,
                                        double *y_bar);
    HPRQPStatus (*update_sigma)(HPRQPSolverBackendContext *context,
                                HPRQPRestart *restart_info,
                                const HPRQPResiduals *residuals);
    HPRQPStatus (*restart_apply)(HPRQPSolverBackendContext *context,
                                 HPRQPRestart *restart_info);
    double (*compute_m_norm)(HPRQPSolverBackendContext *context);
    HPRQPStatus (*iteration_step)(HPRQPSolverBackendContext *context,
                                  size_t inner_iter);
    HPRQPStatus (*iteration_batch)(HPRQPSolverBackendContext *context,
                                   size_t inner_iter,
                                   size_t batch_count);
} HPRQPSolverBackendOps;

HPRQPStatus solve_with_backend(HPRQPSolverBackendContext *context,
                               const HPRQPSolverBackendOps *ops,
                               const char *mode_label,
                               HPRQPResults *results);

#ifdef __cplusplus
}
#endif

#endif
