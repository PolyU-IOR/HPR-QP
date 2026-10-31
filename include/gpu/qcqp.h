#ifndef HPRQP_GPU_QCQP_H
#define HPRQP_GPU_QCQP_H

#include "gpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus gpu_qcqp_setup_majorizer(HPRQPGPUWorkspace *workspace,
                                     const HPRQPGPUProblem *problem,
                                     const HPRQPParams *params);

HPRQPStatus gpu_qcqp_refresh_diag_bq_metric(
    HPRQPGPUWorkspace *workspace,
    double base_cone_metric,
    double base_w_metric);

HPRQPStatus gpu_qcqp_diag_bq_metric_energies(
    HPRQPGPUWorkspace *workspace,
    double *cone_energy,
    double *w_energy);

HPRQPStatus gpu_qcqp_apply_bt(const HPRQPGPUWorkspace *workspace,
                              const HPRQPGPUProblem *problem,
                              const double *dual,
                              double *output);

HPRQPStatus gpu_qcqp_compute_residual_vectors(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem);

HPRQPStatus gpu_qcqp_compute_metric_energy(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem,
    const double *dual_delta);

HPRQPStatus gpu_qcqp_displacement_theta_stats(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem,
    const double *dual_delta,
    double *theta_min,
    double *theta_median,
    double *theta_geomean,
    double *theta_max);

HPRQPStatus gpu_qcqp_iteration_step_parallel(HPRQPGPUWorkspace *workspace,
                                             const HPRQPGPUProblem *problem,
                                             size_t inner_iter);

#ifdef __cplusplus
}
#endif

#endif
