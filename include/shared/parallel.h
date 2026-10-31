#ifndef HPRQP_PARALLEL_H
#define HPRQP_PARALLEL_H

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HPRQP_SECOND_MOMENT_DELTA_TOLERANCE 1e-3

HPRQPStatus parallel_validate_params(const HPRQPParams *params);

double parallel_select_tau(const HPRQPParams *params,
                           double lambda_max_A,
                           double lambda_max_Q);

double parallel_normalized_coupling(double lambda_max_AQA,
                                    double lambda_max_A,
                                    double lambda_max_Q);

double parallel_second_moment_delta(double lambda_max_Q,
                                    double lambda_max_AQA,
                                    double lambda_max_AQ2A);

HPRQPParallelMajorizerMode parallel_select_moment_majorizer(
    double lambda_max_Q,
    double lambda_max_AQA,
    double lambda_max_AQ2A,
    double *delta_2);

double parallel_select_two_moment_tau(const HPRQPParams *params,
                                      double lambda_max_A,
                                      double lambda_max_Q,
                                      double lambda_max_AQA,
                                      double lambda_max_AQ2A,
                                      double y_displacement_sq,
                                      double w_q_displacement);

HPRQPStatus parallel_compute_coefficients(size_t m,
                                          double sigma,
                                          double lambda_max_A,
                                          double lambda_max_Q,
                                          double lambda_max_AQA,
                                          double lambda_max_AQ2A,
                                          double zeta_AQ,
                                          double tau,
                                          HPRQPParallelMajorizerMode majorizer_mode,
                                          double *d_y,
                                          double *d_w,
                                          double *y_rho,
                                          double *y_inv,
                                          double *w_gamma,
                                          double *w_inv);

#ifdef __cplusplus
}
#endif

#endif
