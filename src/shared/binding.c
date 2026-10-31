#include "hprqp_binding.h"

#include "hprqp.h"

#include <stdlib.h>

struct HPRQPBindingModel {
    HPRQPModel model;
};

struct HPRQPBindingParams {
    HPRQPParams params;
};

struct HPRQPBindingResults {
    HPRQPResults results;
    size_t n;
    size_t m;
    HPRQPStatus status;
};

const char *hprqp_status_name(HPRQPStatus status)
{
    switch (status) {
    case HPRQP_STATUS_OK:
        return "OK";
    case HPRQP_STATUS_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case HPRQP_STATUS_ALLOCATION_FAILED:
        return "ALLOCATION_FAILED";
    case HPRQP_STATUS_NOT_IMPLEMENTED:
        return "NOT_IMPLEMENTED";
    case HPRQP_STATUS_MAX_ITER:
        return "MAX_ITER";
    case HPRQP_STATUS_TIME_LIMIT:
        return "TIME_LIMIT";
    case HPRQP_STATUS_OPTIMAL:
        return "OPTIMAL";
    case HPRQP_STATUS_CONTINUE:
        return "CONTINUE";
    default:
        return "UNKNOWN";
    }
}

int hprqp_gpu_backend_available(void)
{
    return gpu_backend_available();
}

static HPRQPStatus init_csr_view(HPRQPMatrixCSR *matrix,
                                 size_t nrow,
                                 size_t ncol,
                                 size_t nnz,
                                 const size_t *row_ptr,
                                 const size_t *col_ind,
                                 const double *values,
                                 size_t *empty_index,
                                 double *empty_value)
{
    if (matrix == NULL || empty_index == NULL || empty_value == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if ((nrow > 0 && row_ptr == NULL) || (nnz > 0 && (col_ind == NULL || values == NULL))) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    matrix->nrow = nrow;
    matrix->ncol = ncol;
    matrix->nnz = nnz;
    matrix->row_ptr = (size_t *)(row_ptr != NULL ? row_ptr : empty_index);
    matrix->col_ind = (size_t *)(col_ind != NULL ? col_ind : empty_index);
    matrix->values = (double *)(values != NULL ? values : empty_value);
    matrix->owns_data = 0;
    return HPRQP_STATUS_OK;
}

static void reset_binding_results(HPRQPBindingResults *results)
{
    if (results == NULL) {
        return;
    }

    results_free(&results->results);
    results_init(&results->results);
    results->n = 0;
    results->m = 0;
    results->status = HPRQP_STATUS_OK;
}

#define HPRQP_BINDING_PARAM_SETTER(name, field, type) \
    HPRQPStatus hprqp_params_set_##name(HPRQPBindingParams *params, type value) \
    { \
        if (params == NULL) { \
            return HPRQP_STATUS_INVALID_ARGUMENT; \
        } \
        params->params.field = value; \
        return HPRQP_STATUS_OK; \
    }

#define HPRQP_BINDING_RESULTS_SIZE_GETTER(name, field) \
    size_t hprqp_results_##name(const HPRQPBindingResults *results) \
    { \
        return results != NULL ? results->field : 0; \
    }

#define HPRQP_BINDING_RESULTS_SCALAR_GETTER(name, field) \
    double hprqp_results_##name(const HPRQPBindingResults *results) \
    { \
        return results != NULL ? results->results.field : 0.0; \
    }

HPRQPStatus hprqp_model_create(HPRQPBindingModel **out_model)
{
    HPRQPBindingModel *model;

    if (out_model == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    model = (HPRQPBindingModel *)malloc(sizeof(*model));
    if (model == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    model_init(&model->model);
    *out_model = model;
    return HPRQP_STATUS_OK;
}

void hprqp_model_destroy(HPRQPBindingModel *model)
{
    if (model == NULL) {
        return;
    }

    model_free(&model->model);
    free(model);
}

HPRQPStatus hprqp_model_read_from_mps(HPRQPBindingModel *model, const char *mps_path)
{
    if (model == NULL || mps_path == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    return build_from_mps(mps_path, &model->model);
}

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
                                            double obj_constant)
{
    HPRQPMatrixCSR q_matrix;
    HPRQPMatrixCSR a_matrix;
    HPRQPStatus status;
    size_t q_empty_index = 0;
    size_t a_empty_index = 0;
    double q_empty_value = 0.0;
    double a_empty_value = 0.0;

    if (model == NULL || c == NULL || l == NULL || u == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = init_csr_view(&q_matrix,
                           q_nrow,
                           q_ncol,
                           q_nnz,
                           q_row_ptr,
                           q_col_ind,
                           q_values,
                           &q_empty_index,
                           &q_empty_value);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = init_csr_view(&a_matrix,
                           a_nrow,
                           a_ncol,
                           a_nnz,
                           a_row_ptr,
                           a_col_ind,
                           a_values,
                           &a_empty_index,
                           &a_empty_value);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    return build_from_QAbc(&q_matrix, &a_matrix, c, AL, AU, l, u, obj_constant, &model->model);
}

size_t hprqp_model_num_variables(const HPRQPBindingModel *model)
{
    return model != NULL ? model->model.Q.ncol : 0;
}

size_t hprqp_model_num_constraints(const HPRQPBindingModel *model)
{
    return model != NULL ? model->model.A.nrow : 0;
}

HPRQPStatus hprqp_params_create(HPRQPBindingParams **out_params)
{
    HPRQPBindingParams *params;

    if (out_params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    params = (HPRQPBindingParams *)malloc(sizeof(*params));
    if (params == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    params_set_defaults(&params->params);
    *out_params = params;
    return HPRQP_STATUS_OK;
}

void hprqp_params_destroy(HPRQPBindingParams *params)
{
    free(params);
}

HPRQPStatus hprqp_params_set_defaults(HPRQPBindingParams *params)
{
    if (params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    params_set_defaults(&params->params);
    return HPRQP_STATUS_OK;
}

HPRQP_BINDING_PARAM_SETTER(stoptol, stoptol, double)
HPRQP_BINDING_PARAM_SETTER(max_iter, max_iter, size_t)
HPRQP_BINDING_PARAM_SETTER(use_gpu, use_gpu, int)
HPRQP_BINDING_PARAM_SETTER(device_number, device_number, int)
HPRQP_BINDING_PARAM_SETTER(time_limit, time_limit, double)
HPRQP_BINDING_PARAM_SETTER(check_iter, check_iter, size_t)
HPRQP_BINDING_PARAM_SETTER(print_frequency, print_frequency, int)
HPRQP_BINDING_PARAM_SETTER(use_scaling, use_scaling, int)
HPRQP_BINDING_PARAM_SETTER(use_cr_scaling, use_cr_scaling, int)
HPRQP_BINDING_PARAM_SETTER(cr_iterations, cr_iterations, size_t)
HPRQP_BINDING_PARAM_SETTER(verbose, verbose, int)
HPRQP_BINDING_PARAM_SETTER(spmv_mode_q, spmv_mode_q, HPRQPSpmvMode)
HPRQP_BINDING_PARAM_SETTER(spmv_mode_a, spmv_mode_a, HPRQPSpmvMode)
HPRQP_BINDING_PARAM_SETTER(method, method, HPRQPMethod)
HPRQP_BINDING_PARAM_SETTER(parallel_majorizer_mode, parallel_majorizer_mode, HPRQPParallelMajorizerMode)
HPRQP_BINDING_PARAM_SETTER(tau_mode, tau_mode, HPRQPTauMode)
HPRQP_BINDING_PARAM_SETTER(tau, tau, double)
HPRQP_BINDING_PARAM_SETTER(tau_min, tau_min, double)
HPRQP_BINDING_PARAM_SETTER(tau_max, tau_max, double)
HPRQP_BINDING_PARAM_SETTER(sigma_fixed, sigma_fixed, int)

HPRQPStatus hprqp_results_create(HPRQPBindingResults **out_results)
{
    HPRQPBindingResults *results;

    if (out_results == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    results = (HPRQPBindingResults *)malloc(sizeof(*results));
    if (results == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    results_init(&results->results);
    results->n = 0;
    results->m = 0;
    results->status = HPRQP_STATUS_OK;
    *out_results = results;
    return HPRQP_STATUS_OK;
}

void hprqp_results_destroy(HPRQPBindingResults *results)
{
    if (results == NULL) {
        return;
    }

    results_free(&results->results);
    free(results);
}

HPRQPStatus hprqp_optimize(const HPRQPBindingModel *model,
                           const HPRQPBindingParams *params,
                           HPRQPBindingResults *results)
{
    if (model == NULL || params == NULL || results == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    reset_binding_results(results);
    results->n = model->model.Q.ncol;
    results->m = model->model.A.nrow;
    results->status = solve(&model->model, &params->params, &results->results);
    return results->status;
}

HPRQPStatus hprqp_results_status(const HPRQPBindingResults *results)
{
    return results != NULL ? results->status : HPRQP_STATUS_INVALID_ARGUMENT;
}

HPRQP_BINDING_RESULTS_SIZE_GETTER(num_variables, n)
HPRQP_BINDING_RESULTS_SIZE_GETTER(num_constraints, m)
HPRQP_BINDING_RESULTS_SIZE_GETTER(iter, results.iter)
HPRQP_BINDING_RESULTS_SIZE_GETTER(iter_4, results.iter_4)
HPRQP_BINDING_RESULTS_SIZE_GETTER(iter_6, results.iter_6)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(time, time)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(time_4, time_4)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(time_6, time_6)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(objective, primal_obj)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(residual, residuals)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(gap, gap)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(lambda_max_A, lambda_max_A)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(lambda_max_Q, lambda_max_Q)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(lambda_max_AQA, lambda_max_AQA)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(lambda_max_AQ2A, lambda_max_AQ2A)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(zeta_AQ, zeta_AQ)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(tau, tau)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(d_y, d_y)
HPRQP_BINDING_RESULTS_SCALAR_GETTER(d_w, d_w)

const double *hprqp_results_x_data(const HPRQPBindingResults *results)
{
    return results != NULL ? results->results.x : NULL;
}

const double *hprqp_results_y_data(const HPRQPBindingResults *results)
{
    return results != NULL ? results->results.y : NULL;
}

const double *hprqp_results_z_data(const HPRQPBindingResults *results)
{
    return results != NULL ? results->results.z : NULL;
}

const double *hprqp_results_w_data(const HPRQPBindingResults *results)
{
    return results != NULL ? results->results.w : NULL;
}
