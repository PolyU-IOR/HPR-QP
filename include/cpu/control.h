#ifndef HPRQP_CONTROL_H
#define HPRQP_CONTROL_H

#include <stddef.h>

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HPRQPMNormTerms {
    double dx_sq;
    double dqw_dx;
    double dy_sq;
    double dqw_atdy;
    double atdy_dx;
    double dqw_sq_or_dw_dqw;
    double atdy_qatdy;
    double m1;
    double m2;
    double m3;
    double total;
} HPRQPMNormTerms;

void restart_init(HPRQPRestart *restart_info);
double compute_m_norm(HPRQPWorkspace *workspace);
HPRQPStatus compute_m_norm_terms(HPRQPWorkspace *workspace, HPRQPMNormTerms *terms);
void restart_check(HPRQPRestart *restart_info, size_t iter, size_t check_iter, double sigma);
void restart_check_socp(HPRQPRestart *restart_info,
                        size_t iter,
                        size_t check_iter,
                        double sigma,
                        double current_kkt);
void restart_post_iteration(HPRQPRestart *restart_info,
                            size_t iter,
                            size_t check_iter,
                            double gap);
HPRQPStatus restart_post_metric(HPRQPRestart *restart_info,
                                HPRQPWorkspace *workspace,
                                size_t iter,
                                size_t check_iter,
                                double gap);
HPRQPStatus restart_post_metric_socp(HPRQPRestart *restart_info,
                                     HPRQPWorkspace *workspace,
                                     size_t iter,
                                     size_t check_iter,
                                     double gap);
HPRQPStatus sigma_update(const HPRQPParams *params,
                         HPRQPRestart *restart_info,
                         HPRQPWorkspace *workspace,
                         const HPRQPResiduals *residuals);
HPRQPStatus restart_apply(HPRQPRestart *restart_info, HPRQPWorkspace *workspace);
double compute_m_norm_parallel(HPRQPWorkspace *workspace);
HPRQPStatus parallel_update_metric(const HPRQPParams *params,
                                   HPRQPRestart *restart_info,
                                   HPRQPWorkspace *workspace,
                                   const HPRQPResiduals *residuals);
HPRQPStatus restart_apply_parallel(HPRQPRestart *restart_info,
                                   HPRQPWorkspace *workspace);

#ifdef __cplusplus
}
#endif

#endif
