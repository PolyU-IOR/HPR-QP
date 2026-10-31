#ifndef HPRQP_BINDING_H
#define HPRQP_BINDING_H

#include <stddef.h>

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HPRQPBindingModel HPRQPBindingModel;
typedef struct HPRQPBindingParams HPRQPBindingParams;
typedef struct HPRQPBindingResults HPRQPBindingResults;

const char *hprqp_status_name(HPRQPStatus status);
int hprqp_gpu_backend_available(void);

HPRQPStatus hprqp_model_create(HPRQPBindingModel **out_model);
void hprqp_model_destroy(HPRQPBindingModel *model);
HPRQPStatus hprqp_model_read_from_mps(HPRQPBindingModel *model, const char *mps_path);
HPRQPStatus hprqp_model_build_from_qabc_csr(HPRQPBindingModel *model,
                                            size_t q_nrow,
                                            size_t q_ncol,
                                            size_t q_nnz,
                                            const size_t *q_row_ptr,
                                            const size_t *q_col_ind,
                                            const double *q_values,
                                            size_t a_nrow,
                                            size_t a_ncol,
                                            size_t a_nnz,
                                            const size_t *a_row_ptr,
                                            const size_t *a_col_ind,
                                            const double *a_values,
                                            const double *c,
                                            const double *AL,
                                            const double *AU,
                                            const double *l,
                                            const double *u,
                                            double obj_constant);
size_t hprqp_model_num_variables(const HPRQPBindingModel *model);
size_t hprqp_model_num_constraints(const HPRQPBindingModel *model);

HPRQPStatus hprqp_params_create(HPRQPBindingParams **out_params);
void hprqp_params_destroy(HPRQPBindingParams *params);
HPRQPStatus hprqp_params_set_defaults(HPRQPBindingParams *params);
HPRQPStatus hprqp_params_set_stoptol(HPRQPBindingParams *params, double value);
HPRQPStatus hprqp_params_set_max_iter(HPRQPBindingParams *params, size_t value);
HPRQPStatus hprqp_params_set_use_gpu(HPRQPBindingParams *params, int value);
HPRQPStatus hprqp_params_set_device_number(HPRQPBindingParams *params, int value);
HPRQPStatus hprqp_params_set_time_limit(HPRQPBindingParams *params, double value);
HPRQPStatus hprqp_params_set_check_iter(HPRQPBindingParams *params, size_t value);
HPRQPStatus hprqp_params_set_print_frequency(HPRQPBindingParams *params, int value);
HPRQPStatus hprqp_params_set_use_scaling(HPRQPBindingParams *params, int value);
HPRQPStatus hprqp_params_set_use_cr_scaling(HPRQPBindingParams *params, int value);
HPRQPStatus hprqp_params_set_cr_iterations(HPRQPBindingParams *params, size_t value);
HPRQPStatus hprqp_params_set_verbose(HPRQPBindingParams *params, int value);
HPRQPStatus hprqp_params_set_spmv_mode_q(HPRQPBindingParams *params, HPRQPSpmvMode value);
HPRQPStatus hprqp_params_set_spmv_mode_a(HPRQPBindingParams *params, HPRQPSpmvMode value);
HPRQPStatus hprqp_params_set_method(HPRQPBindingParams *params, HPRQPMethod value);
HPRQPStatus hprqp_params_set_parallel_majorizer_mode(HPRQPBindingParams *params,
                                                     HPRQPParallelMajorizerMode value);
HPRQPStatus hprqp_params_set_tau_mode(HPRQPBindingParams *params, HPRQPTauMode value);
HPRQPStatus hprqp_params_set_tau(HPRQPBindingParams *params, double value);
HPRQPStatus hprqp_params_set_tau_min(HPRQPBindingParams *params, double value);
HPRQPStatus hprqp_params_set_tau_max(HPRQPBindingParams *params, double value);
HPRQPStatus hprqp_params_set_sigma_fixed(HPRQPBindingParams *params, int value);

HPRQPStatus hprqp_results_create(HPRQPBindingResults **out_results);
void hprqp_results_destroy(HPRQPBindingResults *results);
HPRQPStatus hprqp_optimize(const HPRQPBindingModel *model,
                           const HPRQPBindingParams *params,
                           HPRQPBindingResults *results);
HPRQPStatus hprqp_results_status(const HPRQPBindingResults *results);
size_t hprqp_results_num_variables(const HPRQPBindingResults *results);
size_t hprqp_results_num_constraints(const HPRQPBindingResults *results);
size_t hprqp_results_iter(const HPRQPBindingResults *results);
size_t hprqp_results_iter_4(const HPRQPBindingResults *results);
size_t hprqp_results_iter_6(const HPRQPBindingResults *results);
double hprqp_results_time(const HPRQPBindingResults *results);
double hprqp_results_time_4(const HPRQPBindingResults *results);
double hprqp_results_time_6(const HPRQPBindingResults *results);
double hprqp_results_objective(const HPRQPBindingResults *results);
double hprqp_results_residual(const HPRQPBindingResults *results);
double hprqp_results_gap(const HPRQPBindingResults *results);
double hprqp_results_lambda_max_A(const HPRQPBindingResults *results);
double hprqp_results_lambda_max_Q(const HPRQPBindingResults *results);
double hprqp_results_lambda_max_AQA(const HPRQPBindingResults *results);
double hprqp_results_lambda_max_AQ2A(const HPRQPBindingResults *results);
double hprqp_results_zeta_AQ(const HPRQPBindingResults *results);
double hprqp_results_tau(const HPRQPBindingResults *results);
double hprqp_results_d_y(const HPRQPBindingResults *results);
double hprqp_results_d_w(const HPRQPBindingResults *results);
const double *hprqp_results_x_data(const HPRQPBindingResults *results);
const double *hprqp_results_y_data(const HPRQPBindingResults *results);
const double *hprqp_results_z_data(const HPRQPBindingResults *results);
const double *hprqp_results_w_data(const HPRQPBindingResults *results);

#ifdef __cplusplus
}
#endif

#endif
