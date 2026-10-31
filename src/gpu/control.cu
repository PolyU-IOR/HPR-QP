#include "cuda_common.h"

#include "gpu/qcqp.h"
#include "shared/parallel.h"

#include <cfloat>
#include <chrono>
#include <cstdio>
#include <cstdlib>

enum {
    HPRQP_GPU_FUSED_METRIC_THREADS = 256,
    HPRQP_GPU_FUSED_METRIC_MAX_LENGTH = 32768
};

__global__ static void gpu_parallel_fused_metric_stats_kernel(
    double *output,
    const double *dy,
    const double *dw,
    const double *dqw,
    const double *atdy,
    const double *dx,
    size_t m,
    size_t n)
{
    __shared__ double partial[5][HPRQP_GPU_FUSED_METRIC_THREADS];
    const unsigned int tid = threadIdx.x;
    double local_dy_sq = 0.0;
    double local_dw_dqw = 0.0;
    double local_atdy_dx = 0.0;
    double local_dqw_dx = 0.0;
    double local_dx_sq = 0.0;

    for (size_t i = (size_t)tid; i < m;
         i += (size_t)HPRQP_GPU_FUSED_METRIC_THREADS) {
        local_dy_sq += dy[i] * dy[i];
    }
    for (size_t i = (size_t)tid; i < n;
         i += (size_t)HPRQP_GPU_FUSED_METRIC_THREADS) {
        const double dx_i = dx[i];
        const double dqw_i = dqw[i];
        local_dw_dqw += dw[i] * dqw_i;
        local_atdy_dx += atdy[i] * dx_i;
        local_dqw_dx += dqw_i * dx_i;
        local_dx_sq += dx_i * dx_i;
    }

    partial[0][tid] = local_dy_sq;
    partial[1][tid] = local_dw_dqw;
    partial[2][tid] = local_atdy_dx;
    partial[3][tid] = local_dqw_dx;
    partial[4][tid] = local_dx_sq;
    __syncthreads();

    for (unsigned int stride = HPRQP_GPU_FUSED_METRIC_THREADS / 2U;
         stride > 0U;
         stride >>= 1U) {
        if (tid < stride) {
            partial[0][tid] += partial[0][tid + stride];
            partial[1][tid] += partial[1][tid + stride];
            partial[2][tid] += partial[2][tid + stride];
            partial[3][tid] += partial[3][tid + stride];
            partial[4][tid] += partial[4][tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0U) {
        for (unsigned int slot = 0U; slot < 5U; ++slot) {
            output[slot] = partial[slot][0];
        }
    }
}

__global__ static void gpu_bound_measure_kernel(double *output,
                                                const double *al,
                                                const double *au,
                                                size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double lower;
    double upper;

    if (i >= length) {
        return;
    }

    lower = isinf(al[i]) ? 0.0 : fabs(al[i]);
    upper = isinf(au[i]) ? 0.0 : fabs(au[i]);
    output[i] = lower > upper ? lower : upper;
}

__global__ static void gpu_free_variable_mask_kernel(double *output,
                                                     const double *l,
                                                     const double *u,
                                                     size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    output[i] = (isinf(l[i]) && isinf(u[i])) ? 1.0 : 0.0;
}

__global__ static void gpu_pointwise_multiply_kernel(double *output,
                                                     const double *lhs,
                                                     const double *rhs,
                                                     size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    output[i] = lhs[i] * rhs[i];
}

__global__ static void gpu_vector_diff_kernel(double *output,
                                              const double *lhs,
                                              const double *rhs,
                                              size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    output[i] = lhs[i] - rhs[i];
}

__global__ static void gpu_sigma_diag_eval_kernel(double *output,
                                                  const double *diag_q,
                                                  const double *q_atdy,
                                                  double sigma,
                                                  size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    output[i] = q_atdy[i] / (1.0 + sigma * diag_q[i]);
}

__global__ static void gpu_update_diag_factors_kernel(double *fact1,
                                                      double *fact2,
                                                      double *fact,
                                                      double *fact_m,
                                                      const double *diag_q,
                                                      double sigma,
                                                      size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double temp = 1.0 + sigma * diag_q[i];
    double inv = 1.0 / temp;
    fact2[i] = inv;
    fact[i] = sigma * inv;
    fact1[i] = sigma * diag_q[i] * inv;
    fact_m[i] = sigma * sigma * inv;
}

static HPRQPStatus gpu_compute_bound_norm2_local(cublasHandle_t handle,
                                                 HPRQPGPUWorkspace *workspace,
                                                 const HPRQPGPUVector *al,
                                                 const HPRQPGPUVector *au,
                                                 double *result)
{
    HPRQPStatus status;

    if (result == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->m == 0) {
        *result = 0.0;
        return HPRQP_STATUS_OK;
    }

    gpu_bound_measure_kernel<<<workspace->blocks_m, workspace->threads_m>>>(workspace->reduce_tmp1.values,
                                                                             al->values,
                                                                             au->values,
                                                                             workspace->m);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    return gpu_raw_norm2(handle, workspace->m, workspace->reduce_tmp1.values, result);
}

static HPRQPStatus gpu_compute_noC_local(cublasHandle_t handle,
                                         HPRQPGPUWorkspace *workspace,
                                         const HPRQPGPUProblem *model,
                                         int *noC)
{
    HPRQPStatus status;
    double free_count = 0.0;

    if (noC == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_free_variable_mask_kernel<<<workspace->blocks_n, workspace->threads_n>>>(workspace->reduce_tmp1.values,
                                                                                  model->l.values,
                                                                                  model->u.values,
                                                                                  workspace->n);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = gpu_raw_dot(handle,
                         workspace->n,
                         workspace->reduce_tmp1.values,
                         workspace->reduce_tmp1.values,
                         &free_count);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    *noC = (5.0 * free_count > 4.0 * (double)workspace->n) ? 1 : 0;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus refresh_parallel_gpu_coefficients(HPRQPGPUWorkspace *workspace)
{
    if (workspace->p > 0U) {
        HPRQPStatus status;
        workspace->parallel_y_rho = workspace->empty_A
                                        ? 0.0
                                        : workspace->sigma *
                                              (workspace->qcqp_scalar_f
                                                   ? workspace->d_y *
                                                         workspace->lambda_F
                                                   : workspace->d_y *
                                                         workspace->lambda_max_A);
        workspace->parallel_y_inv = workspace->empty_A
                                        ? 0.0
                                        : 1.0 / workspace->parallel_y_rho;
        if (workspace->qcqp_diag_bq) {
            status = gpu_qcqp_refresh_diag_bq_metric(
                workspace,
                workspace->qcqp_diag_bq_base_s,
                workspace->qcqp_diag_bq_base_w);
            return status;
        }
        workspace->parallel_w_gamma = workspace->sigma * workspace->d_w *
                                      workspace->lambda_max_Q;
        workspace->parallel_w_inv =
            1.0 / (1.0 + workspace->parallel_w_gamma);
        return HPRQP_STATUS_OK;
    }
    return parallel_compute_coefficients(
        workspace->m,
        workspace->sigma,
        workspace->lambda_max_A,
        workspace->lambda_max_Q,
        workspace->lambda_max_AQA,
        workspace->lambda_max_AQ2A,
        workspace->zeta_AQ,
        workspace->tau,
        workspace->parallel_majorizer_mode,
        &workspace->d_y,
        &workspace->d_w,
        &workspace->parallel_y_rho,
        &workspace->parallel_y_inv,
        &workspace->parallel_w_gamma,
        &workspace->parallel_w_inv);
}

static int gpu_scalar_changed(double old_value, double new_value)
{
    double tolerance = 8.0 * DBL_EPSILON * fmax(1.0, fabs(old_value));
    return fabs(old_value - new_value) > tolerance;
}

static HPRQPStatus gpu_parallel_metric_sq_checked(
    const HPRQPGPUWorkspace *workspace,
    double term_y,
    double term_w,
    double term_cross,
    double term_x,
    double *metric_sq)
{
    double value;
    double scale;
    double roundoff_floor;

    if (workspace == NULL || metric_sq == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    value = term_y + term_w + term_cross + term_x;
    scale = fabs(term_y) + fabs(term_w) + fabs(term_cross) + fabs(term_x);
    roundoff_floor = workspace->metric_negative_tolerance * fmax(1.0, scale);
    if (!isfinite(value) || value < -roundoff_floor) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *metric_sq = value > 0.0 ? value : 0.0;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_setup(HPRQPGPUWorkspace *workspace,
                                const HPRQPGPUProblem *model,
                                const HPRQPParams *params)
{
    cublasHandle_t handle;
    HPRQPStatus status;
    double norm_b = 0.0;
    double norm_c = 0.0;
    const bool profile_setup = std::getenv("HPRQP_PROFILE_SETUP") != nullptr;
    const auto profile_start =
        profile_setup ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{};
    auto profile_last = profile_start;
    double norm_c_seconds = 0.0;
    double norm_b_seconds = 0.0;
    double noc_seconds = 0.0;
    double eigen_seconds = 0.0;

    if (workspace == NULL || model == NULL || params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    handle = gpu_blas_handle(workspace);
    if (handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    workspace->to_check = 1;
    workspace->method = params->method;
    workspace->parallel_majorizer_mode = params->parallel_majorizer_mode;
    workspace->metric_negative_tolerance = params->metric_negative_tolerance;

    status = gpu_raw_norm2(handle, workspace->n, model->c.values, &norm_c);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        norm_c_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_compute_bound_norm2_local(handle,
                                           workspace,
                                           &model->AL,
                                           &model->AU,
                                           &norm_b);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        norm_b_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (norm_b > 1e-16 && norm_c > 1e-16 && norm_b < 1e16 && norm_c < 1e16) {
        workspace->sigma = norm_b / norm_c;
    } else {
        workspace->sigma = 1.0;
    }
    if (params->initial_sigma > 0.0) {
        /* The b/c transformation uses x_original = b_scale * x_scaled and
         * scales the objective by 1 / (b_scale * c_scale).  Therefore sigma
         * lives in scaled coordinates as (c_scale / b_scale) * sigma_user.
         * Without this conversion, an equivalent global change of units
         * changes the HPR trajectory. */
        workspace->sigma = params->initial_sigma *
                           workspace->scale_c_scale /
                           workspace->scale_b_scale;
    }

    if (workspace->sigma <= 0.0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_compute_noC_local(handle, workspace, model, &workspace->noC);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        noc_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = gpu_estimate_eigenvalues(workspace, model, params);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        eigen_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (workspace->method == HPRQP_METHOD_PARALLEL) {
        workspace->tau = parallel_select_tau(params,
                                             workspace->lambda_max_A,
                                             workspace->lambda_max_Q);
        workspace->tau_initial = workspace->tau;
        status = parallel_compute_coefficients(
            workspace->m,
            workspace->sigma,
            workspace->lambda_max_A,
            workspace->lambda_max_Q,
            workspace->lambda_max_AQA,
            workspace->lambda_max_AQ2A,
            workspace->zeta_AQ,
            workspace->tau,
            workspace->parallel_majorizer_mode,
            &workspace->d_y,
            &workspace->d_w,
            &workspace->parallel_y_rho,
            &workspace->parallel_y_inv,
            &workspace->parallel_w_gamma,
            &workspace->parallel_w_inv);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        if (model->p > 0) {
            status = gpu_qcqp_setup_majorizer(workspace, model, params);
            if (status != HPRQP_STATUS_OK) {
                return status;
            }
        }
    } else {
        if (model->p > 0) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
        workspace->tau = NAN;
        workspace->tau_initial = NAN;
        workspace->d_y = NAN;
        workspace->d_w = NAN;
        workspace->parallel_y_rho = NAN;
        workspace->parallel_y_inv = NAN;
        workspace->parallel_w_gamma = NAN;
        workspace->parallel_w_inv = NAN;
    }

    if (workspace->Q_is_diag && workspace->threads_n > 0U) {
        gpu_update_diag_factors_kernel<<<workspace->blocks_n, workspace->threads_n>>>(workspace->fact1.values,
                                                                                       workspace->fact2.values,
                                                                                       workspace->fact.values,
                                                                                       workspace->fact_M.values,
                                                                                       workspace->diag_Q.values,
                                                                                       workspace->sigma,
                                                                                       workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (profile_setup) {
        const auto profile_done = std::chrono::steady_clock::now();
        std::fprintf(stderr,
                     "HPRQP_PROFILE workspace: norm_c=%.6f s, "
                     "norm_b=%.6f s, noC=%.6f s, eigenvalues=%.6f s, "
                     "coefficients=%.6f s, total=%.6f s\n",
                     norm_c_seconds,
                     norm_b_seconds,
                     noc_seconds,
                     eigen_seconds,
                     std::chrono::duration<double>(profile_done - profile_last).count(),
                     std::chrono::duration<double>(profile_done - profile_start).count());
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_compute_m_norm_parallel_value(HPRQPGPUWorkspace *gpu_workspace,
                                              double *out_value)
{
    enum {
        SLOT_DY_SQ = 0,
        SLOT_DW_DQW,
        SLOT_ATDY_DX,
        SLOT_DQW_DX,
        SLOT_DX_SQ,
        SLOT_BTDS_DX,
        SLOT_QC_ENERGY,
        SLOT_COUNT
    };
    cublasHandle_t blas_handle;
    HPRQPStatus status;
    HPRQPStatus restore_status;
    double dy_sq = 0.0;
    double dw_dqw;
    double atdy_dx;
    double dqw_dx;
    double dx_sq;
    double btds_dx = 0.0;
    double qc_energy = 0.0;
    double diag_bq_cone_energy = 0.0;
    double diag_bq_w_energy = 0.0;
    double term_y;
    double term_s;
    double term_w;
    double term_cross;
    double term_x;
    double metric_sq;
    double *device_values;
    double host_values[SLOT_COUNT];

    if (gpu_workspace == NULL || out_value == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;
    device_values = gpu_batch_reduce_values(gpu_workspace);
    if (blas_handle == NULL || !(gpu_workspace->sigma > 0.0)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (device_values == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (gpu_workspace->p == 0U &&
        gpu_workspace->n + gpu_workspace->m <=
        (size_t)HPRQP_GPU_FUSED_METRIC_MAX_LENGTH) {
        gpu_parallel_fused_metric_stats_kernel
            <<<1U, HPRQP_GPU_FUSED_METRIC_THREADS>>>(
                device_values,
                gpu_workspace->dy.values,
                gpu_workspace->dw.values,
                gpu_workspace->dQw.values,
                gpu_workspace->ATdy.values,
                gpu_workspace->dx.values,
                gpu_workspace->empty_A ? 0U : gpu_workspace->m,
                gpu_workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = map_cuda_status(cudaMemcpy(host_values,
                                            device_values,
                                            SLOT_COUNT * sizeof(double),
                                            cudaMemcpyDeviceToHost));
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        goto metric_reductions_complete;
    }
    status = gpu_batch_reduce_begin(blas_handle);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (!gpu_workspace->empty_A) {
        status = gpu_raw_dot_device(blas_handle,
                                    gpu_workspace->m,
                                    gpu_workspace->dy.values,
                                    gpu_workspace->dy.values,
                                    device_values + SLOT_DY_SQ);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    }
    status = gpu_raw_dot_device(blas_handle,
                                gpu_workspace->n,
                                gpu_workspace->dw.values,
                                gpu_workspace->dQw.values,
                                device_values + SLOT_DW_DQW);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_raw_dot_device(blas_handle,
                                gpu_workspace->n,
                                gpu_workspace->ATdy.values,
                                gpu_workspace->dx.values,
                                device_values + SLOT_ATDY_DX);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_raw_dot_device(blas_handle,
                                gpu_workspace->n,
                                gpu_workspace->dQw.values,
                                gpu_workspace->dx.values,
                                device_values + SLOT_DQW_DX);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_raw_dot_device(blas_handle,
                                gpu_workspace->n,
                                gpu_workspace->dx.values,
                                gpu_workspace->dx.values,
                                device_values + SLOT_DX_SQ);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    if (gpu_workspace->p > 0U) {
        status = gpu_raw_dot_device(blas_handle,
                                    gpu_workspace->n,
                                    gpu_workspace->BTds.values,
                                    gpu_workspace->dx.values,
                                    device_values + SLOT_BTDS_DX);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_asum_device(
            blas_handle, gpu_workspace->p,
            gpu_workspace->qc_primal_violation.values,
            device_values + SLOT_QC_ENERGY);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    }
    status = gpu_batch_reduce_copy_to_host(gpu_workspace,
                                           host_values,
                                           SLOT_COUNT);

restore_pointer_mode:
    restore_status = gpu_batch_reduce_end(blas_handle);
    if (status != HPRQP_STATUS_OK) return status;
    if (restore_status != HPRQP_STATUS_OK) return restore_status;

metric_reductions_complete:
    dy_sq = gpu_workspace->empty_A ? 0.0 : host_values[SLOT_DY_SQ];
    dw_dqw = host_values[SLOT_DW_DQW];
    atdy_dx = host_values[SLOT_ATDY_DX];
    dqw_dx = host_values[SLOT_DQW_DX];
    dx_sq = host_values[SLOT_DX_SQ];
    if (gpu_workspace->p > 0U) {
        btds_dx = host_values[SLOT_BTDS_DX];
        qc_energy = host_values[SLOT_QC_ENERGY];
    }
    if (gpu_workspace->qcqp_diag_bq) {
        status = gpu_qcqp_diag_bq_metric_energies(
            gpu_workspace, &diag_bq_cone_energy, &diag_bq_w_energy);
        if (status != HPRQP_STATUS_OK) return status;
    }

    term_y = gpu_workspace->empty_A
                 ? 0.0
                 : gpu_workspace->sigma *
                       (gpu_workspace->qcqp_scalar_f
                            ? gpu_workspace->d_y *
                                  gpu_workspace->lambda_F
                            : gpu_workspace->d_y *
                                  gpu_workspace->lambda_max_A) *
                       dy_sq;
    term_s = gpu_workspace->p > 0U
                 ? gpu_workspace->sigma *
                       (gpu_workspace->qcqp_diag_bq
                            ? diag_bq_cone_energy
                            : gpu_workspace->d_s * qc_energy)
                 : 0.0;
    term_w = gpu_workspace->sigma *
             (gpu_workspace->qcqp_diag_bq
                  ? diag_bq_w_energy
                  : gpu_workspace->d_w *
                        gpu_workspace->lambda_max_Q * dw_dqw);
    term_cross = 2.0 * (atdy_dx + btds_dx - dqw_dx);
    term_x = dx_sq / gpu_workspace->sigma;
    status = gpu_parallel_metric_sq_checked(gpu_workspace,
                                            term_y + term_s,
                                            term_w,
                                            term_cross,
                                            term_x,
                                            &metric_sq);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    *out_value = sqrt(metric_sq);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_compute_m_norm_value(HPRQPGPUWorkspace *gpu_workspace,
                                     const HPRQPGPUProblem *gpu_problem,
                                     double *out_value)
{
    double m1 = 0.0;
    double m2;
    double m3 = 0.0;
    double dx_sq;
    double dqw_dx;

    cublasHandle_t blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;
    if (blas_handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    (void)gpu_vector_dot(blas_handle, &gpu_workspace->dx, &gpu_workspace->dx, &dx_sq);

    m2 = (1.0 / gpu_workspace->sigma) * dx_sq;

    (void)gpu_spmv_apply_q_dw(gpu_workspace, gpu_problem);

    (void)gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->dx, &dqw_dx);

    m2 -= 2.0 * dqw_dx;

    if (!gpu_workspace->empty_A) {
        double dy_sq;
        double dqw_atdy;
        double atdy_dx;

        (void)gpu_vector_dot(blas_handle, &gpu_workspace->dy, &gpu_workspace->dy, &dy_sq);

        m1 = gpu_workspace->sigma * gpu_workspace->lambda_max_A * dy_sq;

        (void)gpu_spmv_apply_at_dy(gpu_workspace, gpu_problem);

        (void)gpu_spmv_apply_q_atdy(gpu_workspace, gpu_problem);

        (void)gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->ATdy, &dqw_atdy);

        m1 -= 2.0 * gpu_workspace->sigma * dqw_atdy;

        (void)gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->dx, &atdy_dx);

        m2 += 2.0 * atdy_dx;

        if (gpu_workspace->Q_is_diag) {
            double dqw_sq;
            double diag_term;
            unsigned int threads_n;
            unsigned int blocks_n;

            threads_n = gpu_workspace->threads_n;
            blocks_n = gpu_workspace->blocks_n;
            gpu_pointwise_multiply_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                   gpu_workspace->ATdy.values,
                                                                   gpu_workspace->fact_M.values,
                                                                   gpu_workspace->n);

            gpu_vector_dot(blas_handle, &gpu_workspace->diag_tmp, &gpu_workspace->QATdy, &diag_term);
            m3 = diag_term;

            gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->dQw, &dqw_sq);
            m1 += gpu_workspace->sigma * dqw_sq;
        } else {
            double atdy_qatdy;
            double dw_dqw;

            gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->QATdy, &atdy_qatdy);
            m3 = (gpu_workspace->sigma * gpu_workspace->sigma) /
                 (1.0 + gpu_workspace->sigma * gpu_workspace->lambda_max_Q) * atdy_qatdy;

            gpu_vector_dot(blas_handle, &gpu_workspace->dw, &gpu_workspace->dQw, &dw_dqw);
            m1 += gpu_workspace->sigma * gpu_workspace->lambda_max_Q * dw_dqw;
        }
    } else if (!gpu_workspace->Q_is_diag) {
        double dw_dqw;

        gpu_vector_dot(blas_handle, &gpu_workspace->dw, &gpu_workspace->dQw, &dw_dqw);
        m1 = gpu_workspace->sigma * gpu_workspace->lambda_max_Q * dw_dqw;
    }

    m2 += m1 > 0.0 ? m1 : 0.0;
    *out_value = sqrt((m2 > 0.0 ? m2 : 0.0) + (m3 > 0.0 ? m3 : 0.0));
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_tau_parallel_local(
    HPRQPGPUWorkspace *gpu_workspace,
    const HPRQPParams *params,
    HPRQPRestart *restart_info,
    int *metric_changed)
{
    cublasHandle_t blas_handle;
    HPRQPStatus status;
    double tau_old;
    double tau_new;

    *metric_changed = 0;
    if (restart_info->restart_flag < 1 || gpu_workspace->empty_A) {
        return HPRQP_STATUS_OK;
    }
    tau_old = gpu_workspace->tau;
    if (params->tau_mode != HPRQP_TAU_MODE_ADAPTIVE_RESTART) {
        tau_new = parallel_select_tau(params,
                                      gpu_workspace->lambda_max_A,
                                      gpu_workspace->lambda_max_Q);
    } else if (gpu_workspace->parallel_majorizer_mode ==
                   HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER &&
               gpu_workspace->zeta_AQ == 0.0) {
        tau_new = parallel_select_tau(params,
                                      gpu_workspace->lambda_max_A,
                                      gpu_workspace->lambda_max_Q);
    } else {
        double dy_sq;
        double dw_dqw;

        blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;
        status = gpu_vector_dot(blas_handle,
                                &gpu_workspace->dy,
                                &gpu_workspace->dy,
                                &dy_sq);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_dot(blas_handle,
                                &gpu_workspace->dw,
                                &gpu_workspace->dQw,
                                &dw_dqw);
        if (status != HPRQP_STATUS_OK) return status;
        if (gpu_workspace->parallel_majorizer_mode ==
            HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT) {
            tau_new = parallel_select_two_moment_tau(
                params,
                gpu_workspace->lambda_max_A,
                gpu_workspace->lambda_max_Q,
                gpu_workspace->lambda_max_AQA,
                gpu_workspace->lambda_max_AQ2A,
                dy_sq,
                dw_dqw);
        } else {
            double y_energy = gpu_workspace->lambda_max_A * dy_sq;
            double w_energy = gpu_workspace->lambda_max_Q * dw_dqw;
            if (!(isfinite(y_energy) &&
                  y_energy > params->parallel_adaptation_energy_floor &&
                  isfinite(w_energy) &&
                  w_energy > params->parallel_adaptation_energy_floor)) {
                tau_new = parallel_select_tau(params,
                                              gpu_workspace->lambda_max_A,
                                              gpu_workspace->lambda_max_Q);
            } else {
                double tau_raw = sqrt(w_energy / y_energy);
                tau_new = isfinite(tau_raw) && tau_raw > 0.0
                              ? clamp_double_local(tau_raw,
                                                   params->tau_min,
                                                   params->tau_max)
                              : parallel_select_tau(params,
                                                    gpu_workspace->lambda_max_A,
                                                    gpu_workspace->lambda_max_Q);
            }
        }
    }

    if (!gpu_scalar_changed(tau_old, tau_new)) {
        return HPRQP_STATUS_OK;
    }
    gpu_workspace->tau = tau_new;
    status = refresh_parallel_gpu_coefficients(gpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace->tau = tau_old;
        return status;
    }
    *metric_changed = !(gpu_workspace->parallel_majorizer_mode ==
                            HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER &&
                        gpu_workspace->zeta_AQ == 0.0);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_joint_f_shape_qcqp_local(
    HPRQPGPUWorkspace *gpu_workspace,
    const HPRQPGPUProblem *gpu_problem,
    const HPRQPParams *params,
    int *metric_changed)
{
    const double tau_old = gpu_workspace->tau_AQ;
    const double d_y_old = gpu_workspace->d_y;
    const double d_s_old = gpu_workspace->d_s;
    const double d_w_old = gpu_workspace->d_w;
    double tau_new = tau_old;
    HPRQPStatus status;

    *metric_changed = 0;
    if (!(gpu_workspace->zeta_AQ > 0.0) ||
        !(gpu_workspace->lambda_max_Q > 0.0)) {
        return HPRQP_STATUS_OK;
    }
    if (params->tau_mode != HPRQP_TAU_MODE_ADAPTIVE_RESTART) {
        tau_new = parallel_select_tau(params,
                                      gpu_workspace->lambda_F,
                                      gpu_workspace->lambda_max_Q);
    } else {
        cublasHandle_t blas_handle =
            (cublasHandle_t)gpu_workspace->blas_handle;
        double dy_sq = 0.0;
        double dw_dqw = 0.0;
        double qc_energy = 0.0;
        double energy_f;
        double energy_w;
        const double floor = params->parallel_adaptation_energy_floor;

        if (blas_handle == NULL) return HPRQP_STATUS_INVALID_ARGUMENT;
        if (!gpu_workspace->empty_A) {
            status = gpu_vector_dot(blas_handle,
                                    &gpu_workspace->dy,
                                    &gpu_workspace->dy,
                                    &dy_sq);
            if (status != HPRQP_STATUS_OK) return status;
        }
        status = gpu_vector_dot(blas_handle,
                                &gpu_workspace->dw,
                                &gpu_workspace->dQw,
                                &dw_dqw);
        if (status != HPRQP_STATUS_OK) return status;
        if (std::getenv("HPRQP_TRACE_QC_THETA_DISP") != nullptr) {
            double theta_min;
            double theta_median;
            double theta_geomean;
            double theta_max;
            status = gpu_qcqp_displacement_theta_stats(
                gpu_workspace, gpu_problem,
                gpu_workspace->qc_dual_delta.values,
                &theta_min, &theta_median, &theta_geomean, &theta_max);
            if (status != HPRQP_STATUS_OK) return status;
            printf("HPRQP_QC_THETA_DISP min=%.12g median=%.12g "
                   "geomean=%.12g max=%.12g\n",
                   theta_min, theta_median, theta_geomean, theta_max);
        }
        status = gpu_qcqp_compute_metric_energy(
            gpu_workspace, gpu_problem,
            gpu_workspace->qc_dual_delta.values);
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cublas_status(cublasDasum(
            blas_handle, (int)gpu_workspace->p,
            gpu_workspace->qc_primal_violation.values, 1,
            &qc_energy));
        if (status != HPRQP_STATUS_OK) return status;

        energy_f = gpu_workspace->lambda_F * dy_sq + qc_energy;
        energy_w = gpu_workspace->lambda_max_Q * dw_dqw;
        if (isfinite(energy_f) && energy_f > floor &&
            isfinite(energy_w) && energy_w > floor) {
            tau_new = clamp_double_local(sqrt(energy_w / energy_f),
                                         1e-2, 1e2);
        }
    }

    gpu_workspace->tau_AQ = tau_new;
    gpu_workspace->tau = tau_new;
    gpu_workspace->d_y = 1.0 + gpu_workspace->zeta_AQ * tau_new;
    gpu_workspace->d_s = gpu_workspace->d_y;
    gpu_workspace->d_w = 1.0 + gpu_workspace->zeta_AQ / tau_new;
    status = refresh_parallel_gpu_coefficients(gpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace->tau_AQ = tau_old;
        gpu_workspace->tau = tau_old;
        gpu_workspace->d_y = d_y_old;
        gpu_workspace->d_s = d_s_old;
        gpu_workspace->d_w = d_w_old;
        (void)refresh_parallel_gpu_coefficients(gpu_workspace);
        return status;
    }
    *metric_changed = gpu_scalar_changed(d_y_old, gpu_workspace->d_y) ||
                      gpu_scalar_changed(d_s_old, gpu_workspace->d_s) ||
                      gpu_scalar_changed(d_w_old, gpu_workspace->d_w);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_pairwise_shapes_qcqp_local(
    HPRQPGPUWorkspace *gpu_workspace,
    const HPRQPGPUProblem *gpu_problem,
    const HPRQPParams *params,
    HPRQPRestart *restart_info,
    int *metric_changed)
{
    double tau_ab_old;
    double tau_aq_old;
    double tau_bq_old;
    double d_y_old;
    double d_s_old;
    double d_w_old;
    double diag_bq_base_s_old;
    double diag_bq_base_w_old;
    double tau_ab_new;
    double tau_aq_new;
    double tau_bq_new;
    double d_y_aq;
    double d_w_aq;
    double dummy_y_rho;
    double dummy_y_inv;
    double dummy_w_gamma;
    double dummy_w_inv;
    HPRQPStatus status;

    *metric_changed = 0;
    if (restart_info->restart_flag < 1) {
        return HPRQP_STATUS_OK;
    }
    if (gpu_workspace->qcqp_scalar_f) {
        return gpu_update_joint_f_shape_qcqp_local(
            gpu_workspace, gpu_problem, params, metric_changed);
    }

    tau_ab_old = gpu_workspace->tau_AB;
    tau_aq_old = gpu_workspace->tau_AQ;
    tau_bq_old = gpu_workspace->tau_BQ;
    d_y_old = gpu_workspace->d_y;
    d_s_old = gpu_workspace->d_s;
    d_w_old = gpu_workspace->d_w;
    diag_bq_base_s_old = gpu_workspace->qcqp_diag_bq_base_s;
    diag_bq_base_w_old = gpu_workspace->qcqp_diag_bq_base_w;
    tau_ab_new = tau_ab_old;
    tau_aq_new = tau_aq_old;
    tau_bq_new = tau_bq_old;

    if (params->tau_mode != HPRQP_TAU_MODE_ADAPTIVE_RESTART) {
        tau_ab_new = 1.0;
        tau_aq_new = parallel_select_tau(params,
                                         gpu_workspace->lambda_max_A,
                                         gpu_workspace->lambda_max_Q);
        tau_bq_new = 1.0;
    } else {
        cublasHandle_t blas_handle =
            (cublasHandle_t)gpu_workspace->blas_handle;
        double dy_sq = 0.0;
        double dw_dqw = 0.0;
        double qc_energy = 0.0;
        double energy_y;
        double energy_s;
        double energy_w;
        const double floor = params->parallel_adaptation_energy_floor;

        if (blas_handle == NULL) return HPRQP_STATUS_INVALID_ARGUMENT;
        if (!gpu_workspace->empty_A) {
            status = gpu_vector_dot(blas_handle,
                                    &gpu_workspace->dy,
                                    &gpu_workspace->dy,
                                    &dy_sq);
            if (status != HPRQP_STATUS_OK) return status;
        }
        status = gpu_vector_dot(blas_handle,
                                &gpu_workspace->dw,
                                &gpu_workspace->dQw,
                                &dw_dqw);
        if (status != HPRQP_STATUS_OK) return status;
        if (std::getenv("HPRQP_TRACE_QC_THETA_DISP") != nullptr) {
            double theta_min;
            double theta_median;
            double theta_geomean;
            double theta_max;
            status = gpu_qcqp_displacement_theta_stats(
                gpu_workspace, gpu_problem,
                gpu_workspace->qc_dual_delta.values,
                &theta_min, &theta_median, &theta_geomean, &theta_max);
            if (status != HPRQP_STATUS_OK) return status;
            printf("HPRQP_QC_THETA_DISP min=%.12g median=%.12g "
                   "geomean=%.12g max=%.12g\n",
                   theta_min, theta_median, theta_geomean, theta_max);
        }
        status = gpu_qcqp_compute_metric_energy(
            gpu_workspace, gpu_problem,
            gpu_workspace->qc_dual_delta.values);
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cublas_status(cublasDasum(
            blas_handle, (int)gpu_workspace->p,
            gpu_workspace->qc_primal_violation.values, 1,
            &qc_energy));
        if (status != HPRQP_STATUS_OK) return status;

        energy_y = gpu_workspace->lambda_max_A * dy_sq;
        energy_s = qc_energy;
        energy_w = gpu_workspace->lambda_max_Q * dw_dqw;
        if (gpu_workspace->zeta_AB > 0.0 &&
            isfinite(energy_y) && energy_y > floor &&
            isfinite(energy_s) && energy_s > floor) {
            tau_ab_new = clamp_double_local(sqrt(energy_s / energy_y),
                                            params->tau_min,
                                            params->tau_max);
        }
        if (gpu_workspace->zeta_AQ > 0.0) {
            if (gpu_workspace->parallel_majorizer_mode ==
                HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT) {
                tau_aq_new = parallel_select_two_moment_tau(
                    params,
                    gpu_workspace->lambda_max_A,
                    gpu_workspace->lambda_max_Q,
                    gpu_workspace->lambda_max_AQA,
                    gpu_workspace->lambda_max_AQ2A,
                    dy_sq,
                    dw_dqw);
            } else if (isfinite(energy_y) && energy_y > floor &&
                       isfinite(energy_w) && energy_w > floor) {
                tau_aq_new = clamp_double_local(sqrt(energy_w / energy_y),
                                                params->tau_min,
                                                params->tau_max);
            }
        }
        if (!gpu_workspace->qcqp_diag_bq &&
            gpu_workspace->zeta_BQ > 0.0 &&
            isfinite(energy_s) && energy_s > floor &&
            isfinite(energy_w) && energy_w > floor) {
            tau_bq_new = clamp_double_local(sqrt(energy_w / energy_s),
                                            params->tau_min, params->tau_max);
        }
    }

    gpu_workspace->tau_AB = tau_ab_new;
    gpu_workspace->tau_AQ = tau_aq_new;
    gpu_workspace->tau_BQ = tau_bq_new;
    gpu_workspace->tau = tau_aq_new;
    status = parallel_compute_coefficients(
        gpu_workspace->m,
        gpu_workspace->sigma,
        gpu_workspace->lambda_max_A,
        gpu_workspace->lambda_max_Q,
        gpu_workspace->lambda_max_AQA,
        gpu_workspace->lambda_max_AQ2A,
        gpu_workspace->zeta_AQ,
        tau_aq_new,
        gpu_workspace->parallel_majorizer_mode,
        &d_y_aq,
        &d_w_aq,
        &dummy_y_rho,
        &dummy_y_inv,
        &dummy_w_gamma,
        &dummy_w_inv);
    if (status != HPRQP_STATUS_OK) return status;
    gpu_workspace->d_y = d_y_aq +
        gpu_workspace->zeta_AB * tau_ab_new;
    if (gpu_workspace->qcqp_diag_bq) {
        gpu_workspace->qcqp_diag_bq_base_s =
            1.0 + gpu_workspace->zeta_AB / tau_ab_new;
        gpu_workspace->qcqp_diag_bq_base_w = d_w_aq;
    } else {
        gpu_workspace->d_s = 1.0 +
            gpu_workspace->zeta_AB / tau_ab_new +
            gpu_workspace->zeta_BQ * tau_bq_new;
        gpu_workspace->d_w = d_w_aq +
            gpu_workspace->zeta_BQ / tau_bq_new;
    }
    status = refresh_parallel_gpu_coefficients(gpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace->tau_AB = tau_ab_old;
        gpu_workspace->tau_AQ = tau_aq_old;
        gpu_workspace->tau_BQ = tau_bq_old;
        gpu_workspace->tau = tau_aq_old;
        gpu_workspace->d_y = d_y_old;
        gpu_workspace->d_s = d_s_old;
        gpu_workspace->d_w = d_w_old;
        gpu_workspace->qcqp_diag_bq_base_s = diag_bq_base_s_old;
        gpu_workspace->qcqp_diag_bq_base_w = diag_bq_base_w_old;
        (void)refresh_parallel_gpu_coefficients(gpu_workspace);
        return status;
    }
    *metric_changed = gpu_scalar_changed(d_y_old, gpu_workspace->d_y) ||
                      gpu_scalar_changed(d_s_old, gpu_workspace->d_s) ||
                      gpu_scalar_changed(d_w_old, gpu_workspace->d_w);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_sigma_parallel_local(
    HPRQPGPUWorkspace *gpu_workspace,
    const HPRQPGPUProblem *gpu_problem,
    const HPRQPParams *params,
    HPRQPRestart *restart_info,
    const HPRQPResiduals *residuals,
    int *sigma_changed)
{
    cublasHandle_t blas_handle;
    HPRQPStatus status;
    double sigma_old;
    double dx_sq;
    double dw_dqw;
    double dqw_dx;
    double dy_sq = 0.0;
    double atdy_dx = 0.0;
    double btds_dx = 0.0;
    double qc_energy = 0.0;
    double diag_bq_cone_energy = 0.0;
    double diag_bq_w_energy = 0.0;
    double a_M;
    double b_M;
    double c_M;
    double metric_sq;
    double sigma_estimation;
    double fact;
    double sigma_cand;
    double temp_1;
    double ratio_infeas_org;
    double kappa;
    double sigma_new;

    *sigma_changed = 0;
    if (params->sigma_fixed || restart_info->restart_flag < 1) {
        return HPRQP_STATUS_OK;
    }

    blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;
    sigma_old = gpu_workspace->sigma;

    gpu_vector_diff_kernel<<<gpu_workspace->blocks_n,
                             gpu_workspace->threads_n>>>(
        gpu_workspace->dx.values,
        gpu_workspace->x_bar.values,
        gpu_workspace->last_x.values,
        gpu_workspace->n);
    gpu_vector_diff_kernel<<<gpu_workspace->blocks_n,
                             gpu_workspace->threads_n>>>(
        gpu_workspace->dw.values,
        gpu_workspace->w_bar.values,
        gpu_workspace->last_w.values,
        gpu_workspace->n);
    gpu_vector_diff_kernel<<<gpu_workspace->blocks_n,
                             gpu_workspace->threads_n>>>(
        gpu_workspace->dQw.values,
        gpu_workspace->Qw_bar.values,
        gpu_workspace->last_Qw.values,
        gpu_workspace->n);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) return status;

    if (!gpu_workspace->empty_A) {
        gpu_vector_diff_kernel<<<gpu_workspace->blocks_m,
                                 gpu_workspace->threads_m>>>(
            gpu_workspace->dy.values,
            gpu_workspace->y_bar.values,
            gpu_workspace->last_y.values,
            gpu_workspace->m);
        gpu_vector_diff_kernel<<<gpu_workspace->blocks_n,
                                 gpu_workspace->threads_n>>>(
            gpu_workspace->ATdy.values,
            gpu_workspace->ATy_bar.values,
            gpu_workspace->last_ATy.values,
            gpu_workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
    } else {
        status = map_cuda_status(cudaMemset(gpu_workspace->ATdy.values,
                                            0,
                                            gpu_workspace->n * sizeof(double)));
        if (status != HPRQP_STATUS_OK) return status;
    }
    if (gpu_workspace->p > 0U) {
        const unsigned int qc_dual_blocks = (unsigned int)(
            (gpu_workspace->qc_cone_dim + gpu_workspace->threads_p - 1U) /
            gpu_workspace->threads_p);
        gpu_vector_diff_kernel<<<qc_dual_blocks,
                                 gpu_workspace->threads_p>>>(
            gpu_workspace->qc_dual_delta.values,
            gpu_workspace->qc_dual_bar.values,
            gpu_workspace->last_qc_dual.values,
            gpu_workspace->qc_cone_dim);
        gpu_vector_diff_kernel<<<gpu_workspace->blocks_n,
                                 gpu_workspace->threads_n>>>(
            gpu_workspace->BTds.values,
            gpu_workspace->BTs_bar.values,
            gpu_workspace->last_BTs.values,
            gpu_workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_qcqp_compute_metric_energy(
            gpu_workspace, gpu_problem,
            gpu_workspace->qc_dual_delta.values);
        if (status != HPRQP_STATUS_OK) return status;
    }

    status = gpu_vector_dot(blas_handle,
                            &gpu_workspace->dx,
                            &gpu_workspace->dx,
                            &dx_sq);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_dot(blas_handle,
                            &gpu_workspace->dw,
                            &gpu_workspace->dQw,
                            &dw_dqw);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_dot(blas_handle,
                            &gpu_workspace->dQw,
                            &gpu_workspace->dx,
                            &dqw_dx);
    if (status != HPRQP_STATUS_OK) return status;
    if (!gpu_workspace->empty_A) {
        status = gpu_vector_dot(blas_handle,
                                &gpu_workspace->dy,
                                &gpu_workspace->dy,
                                &dy_sq);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_vector_dot(blas_handle,
                                &gpu_workspace->ATdy,
                                &gpu_workspace->dx,
                                &atdy_dx);
        if (status != HPRQP_STATUS_OK) return status;
    }
    if (gpu_workspace->p > 0U) {
        status = gpu_vector_dot(blas_handle,
                                &gpu_workspace->BTds,
                                &gpu_workspace->dx,
                                &btds_dx);
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cublas_status(cublasDasum(
            blas_handle, (int)gpu_workspace->p,
            gpu_workspace->qc_primal_violation.values, 1,
            &qc_energy));
        if (status != HPRQP_STATUS_OK) return status;
    }

    if (gpu_workspace->qcqp_diag_bq) {
        status = gpu_qcqp_diag_bq_metric_energies(
            gpu_workspace, &diag_bq_cone_energy, &diag_bq_w_energy);
        if (status != HPRQP_STATUS_OK) return status;
    }

    a_M = gpu_workspace->qcqp_diag_bq
              ? diag_bq_w_energy
              : gpu_workspace->d_w * gpu_workspace->lambda_max_Q * dw_dqw;
    if (!gpu_workspace->empty_A) {
        a_M += (gpu_workspace->qcqp_scalar_f
                    ? gpu_workspace->d_y * gpu_workspace->lambda_F
                    : gpu_workspace->d_y * gpu_workspace->lambda_max_A) *
               dy_sq;
    }
    if (gpu_workspace->p > 0U) {
        a_M += gpu_workspace->qcqp_diag_bq
                   ? diag_bq_cone_energy
                   : gpu_workspace->d_s * qc_energy;
    }
    b_M = dx_sq;
    c_M = atdy_dx + btds_dx - dqw_dx;
    status = gpu_parallel_metric_sq_checked(gpu_workspace,
                                            sigma_old * a_M,
                                            0.0,
                                            2.0 * c_M,
                                            b_M / sigma_old,
                                            &metric_sq);
    if (status != HPRQP_STATUS_OK) return status;
    (void)metric_sq;

    if (!(isfinite(a_M) &&
          a_M > params->parallel_adaptation_energy_floor &&
          isfinite(b_M) &&
          b_M > params->parallel_adaptation_energy_floor)) {
        return HPRQP_STATUS_OK;
    }
    sigma_estimation = sqrt(fmax(b_M, 1e-12) / fmax(a_M, 1e-12));
    if (params->use_socp_sigma_policy) {
        const double eps = 1e-16;
        const double threshold = 1e-8;
        const double imbalance_trigger = 1e6;
        const double correction_ratio_floor = 2.0;
        const double primal_infeas =
            fmax(residuals->err_Rp_org_bar, residuals->err_QC_org_bar);
        const double err_p = fmax(primal_infeas, eps);
        const double err_d = fmax(residuals->err_Rd_org_bar, eps);
        const double err_small = fmin(err_p, err_d);
        const double err_large = fmax(err_p, err_d);
        const double imbalance_ratio = err_large / err_small;
        const int both_finished = err_large <= threshold;
        const int one_side_finished = err_small <= 0.5 * threshold &&
                                      err_large >= 2.0 * threshold;
        const int extreme_imbalance = imbalance_ratio >= imbalance_trigger;
        const int correction_trigger =
            !both_finished && (one_side_finished || extreme_imbalance);
        const int correction_keep =
            !both_finished &&
            (err_small <= 0.8 * threshold ||
             imbalance_ratio >= 0.1 * imbalance_trigger);
        const int current_dir = err_d >= err_p ? 1 : -1;
        const double current_kkt = residuals->KKTx_and_gap_org_bar;
        double best_kkt = restart_info->best_kkt;
        double best_gap = restart_info->best_gap;
        double sigma_ref = restart_info->best_sigma;
        int reference_ok;
        double progress_ratio;
        double theta;
        double beta = 0.0;

        if (!(isfinite(best_kkt) && best_kkt > 0.0)) {
            best_kkt = current_kkt;
            restart_info->best_kkt = current_kkt;
        }
        if (!(isfinite(best_gap) && best_gap > 0.0)) {
            best_gap = restart_info->current_gap;
        }
        if (!(isfinite(sigma_ref) && sigma_ref > 0.0)) {
            sigma_ref = sigma_old;
        }
        reference_ok = current_kkt <= 10.0 * best_kkt &&
                       restart_info->current_gap <= 10.0 * best_gap;
        progress_ratio = sqrt(
            (current_kkt / fmax(best_kkt, eps)) *
            (restart_info->current_gap / fmax(best_gap, eps)));
        progress_ratio = clamp_double_local(progress_ratio, 1.0, 10.0);
        theta = clamp_double_local(
            exp(-0.3 * (progress_ratio - 1.0)), 0.0, 1.0);
        sigma_cand = reference_ok
                         ? exp(theta * log(sigma_estimation) +
                               (1.0 - theta) * log(sigma_ref))
                         : sigma_estimation;

        if (both_finished) {
            restart_info->sigma_correction_active = 0;
            restart_info->sigma_correction_hold = -1;
            restart_info->sigma_correction_dir = 0;
        } else if (correction_trigger ||
                   (restart_info->sigma_correction_active &&
                    correction_keep)) {
            restart_info->sigma_correction_active = 1;
            restart_info->sigma_correction_hold = 0;
            restart_info->sigma_correction_dir = current_dir;
        } else {
            restart_info->sigma_correction_active = 0;
            restart_info->sigma_correction_hold = -1;
            restart_info->sigma_correction_dir = 0;
        }

        if (!both_finished) {
            if (one_side_finished) beta = 0.5;
            if (extreme_imbalance) {
                beta = 1.0;
            } else if (restart_info->sigma_correction_active) {
                beta = 0.5;
            }
        }
        kappa = 1.0;
        if (beta > 0.0) {
            double ratio_for_kappa = err_d / err_p;
            if (restart_info->sigma_correction_active) {
                if (restart_info->sigma_correction_dir > 0) {
                    ratio_for_kappa =
                        fmax(ratio_for_kappa, correction_ratio_floor);
                } else if (restart_info->sigma_correction_dir < 0) {
                    ratio_for_kappa =
                        fmin(ratio_for_kappa,
                             1.0 / correction_ratio_floor);
                }
            }
            kappa = exp(beta * clamp_double_local(
                                   log(ratio_for_kappa),
                                   -log(100.0), log(100.0)));
        }
        sigma_new = sigma_cand * kappa;
        if (restart_info->sigma_correction_active) {
            if (restart_info->sigma_correction_dir > 0) {
                sigma_new = fmax(sigma_new, 1.5 * sigma_old);
            } else if (restart_info->sigma_correction_dir < 0) {
                sigma_new = fmin(sigma_new, sigma_old / 1.5);
            }
        }
        restart_info->best_iter += 1U;
    } else {
        fact = (restart_info->best_gap > 0.0 &&
                isfinite(restart_info->best_gap))
                   ? exp(-0.05 *
                         (restart_info->current_gap /
                          restart_info->best_gap))
                   : 1.0;
        sigma_cand = exp(
            fact * log(sigma_estimation) +
            (1.0 - fact) *
                log(restart_info->best_sigma > 0.0
                        ? restart_info->best_sigma
                        : sigma_old));
        const double primal_infeas =
            fmax(residuals->err_Rp_org_bar, residuals->err_QC_org_bar);
        temp_1 = fmax(fmin(residuals->err_Rd_org_bar,
                           primal_infeas),
                      fmin(residuals->rel_gap_bar,
                           restart_info->current_gap));
        if (primal_infeas > 0.0) {
            ratio_infeas_org = residuals->err_Rd_org_bar /
                               primal_infeas;
        } else if (residuals->err_Rd_org_bar > 0.0) {
            ratio_infeas_org = INFINITY;
        } else {
            ratio_infeas_org = 1.0;
        }
        if (temp_1 > 9e-10) {
            kappa = 1.0;
        } else if (temp_1 > 5e-10) {
            kappa = clamp_double_local(sqrt(ratio_infeas_org),
                                       1e-2, 100.0);
        } else {
            kappa = clamp_double_local(ratio_infeas_org,
                                       1e-2, 100.0);
        }
        sigma_new = kappa * sigma_cand;
    }
    if (gpu_workspace->p > 0U) {
        /* QCQP safeguard: one decade per restart for every sigma policy. */
        sigma_new = clamp_double_local(sigma_new,
                                       0.1 * sigma_old,
                                       10.0 * sigma_old);
    }
    if (!(isfinite(sigma_new) && sigma_new > 0.0)) {
        return HPRQP_STATUS_OK;
    }
    sigma_new = clamp_double_local(sigma_new, 1e-12, 1e12);
    if (fabs(sigma_old - sigma_new) > 1e-15) {
        gpu_workspace->sigma = sigma_new;
        status = refresh_parallel_gpu_coefficients(gpu_workspace);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace->sigma = sigma_old;
            (void)refresh_parallel_gpu_coefficients(gpu_workspace);
            return status;
        }
        *sigma_changed = 1;
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_parallel_update_metric_value(
    HPRQPGPUWorkspace *gpu_workspace,
    const HPRQPGPUProblem *gpu_problem,
    const HPRQPParams *params,
    HPRQPRestart *restart_info,
    const HPRQPResiduals *residuals)
{
    HPRQPStatus status;
    int tau_changed;
    int sigma_changed;

    if (gpu_workspace == NULL || params == NULL || restart_info == NULL ||
        residuals == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (gpu_workspace->p > 0U) {
        status = gpu_update_pairwise_shapes_qcqp_local(
            gpu_workspace, gpu_problem, params, restart_info,
            &tau_changed);
    } else {
        status = gpu_update_tau_parallel_local(gpu_workspace,
                                               params,
                                               restart_info,
                                               &tau_changed);
    }
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_update_sigma_parallel_local(gpu_workspace,
                                             gpu_problem,
                                             params,
                                             restart_info,
                                             residuals,
                                             &sigma_changed);
    if (status != HPRQP_STATUS_OK) return status;
    if (tau_changed || sigma_changed) {
        restart_info->metric_reseed_pending = 1;
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_sigma_update_value(HPRQPGPUWorkspace *gpu_workspace,
                                   const HPRQPGPUProblem *gpu_problem,
                                   const HPRQPParams *params,
                                   const HPRQPRestart *restart_info,
                                   const HPRQPResiduals *residuals,
                                   double *out_sigma)
{
    cublasHandle_t blas_handle;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;
    double sigma_old;
    double a = 0.0;
    double b;
    double sigma_estimation;
    double sigma_cand;
    double fact;
    double temp_1;
    double kappa = 1.0;
    double ratio_infeas_org;
    double c_term = 0.0;
    double d_term = 0.0;
    double dx_sq;
    double dy_sq = 0.0;
    double dqw_atdy = 0.0;
    double dqw_sq_or_dw_dqw = 0.0;

    if (restart_info->restart_flag < 1) {
        *out_sigma = gpu_workspace->sigma;
        return HPRQP_STATUS_OK;
    }

    blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;

    threads_n = gpu_workspace->threads_n;
    blocks_n = gpu_workspace->blocks_n;
    threads_m = gpu_workspace->threads_m;
    blocks_m = gpu_workspace->blocks_m;
    sigma_old = gpu_workspace->sigma;

    gpu_vector_diff_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dx.values,
                                                    gpu_workspace->x_bar.values,
                                                    gpu_workspace->last_x.values,
                                                    gpu_workspace->n);
    gpu_vector_diff_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dw.values,
                                                    gpu_workspace->w_bar.values,
                                                    gpu_workspace->last_w.values,
                                                    gpu_workspace->n);

    gpu_vector_dot(blas_handle, &gpu_workspace->dx, &gpu_workspace->dx, &dx_sq);

    b = dx_sq;

    gpu_spmv_apply_q_dw(gpu_workspace, gpu_problem);

    if (!gpu_workspace->empty_A) {
        gpu_vector_diff_kernel<<<blocks_m, threads_m>>>(gpu_workspace->dy.values,
                                                        gpu_workspace->y_bar.values,
                                                        gpu_workspace->last_y.values,
                                                        gpu_workspace->m);
        gpu_vector_diff_kernel<<<blocks_n, threads_n>>>(gpu_workspace->ATdy.values,
                                                        gpu_workspace->ATy_bar.values,
                                                        gpu_workspace->last_ATy.values,
                                                        gpu_workspace->n);

        gpu_vector_dot(blas_handle, &gpu_workspace->dy, &gpu_workspace->dy, &dy_sq);
        a = gpu_workspace->lambda_max_A * dy_sq;

        gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->ATdy, &dqw_atdy);
        a -= 2.0 * dqw_atdy;

        gpu_spmv_apply_q_atdy(gpu_workspace, gpu_problem);
    }

    if (gpu_workspace->Q_is_diag) {
        gpu_vector_dot(blas_handle, &gpu_workspace->dQw, &gpu_workspace->dQw, &dqw_sq_or_dw_dqw);
        a += dqw_sq_or_dw_dqw;
    } else {
        gpu_vector_dot(blas_handle, &gpu_workspace->dw, &gpu_workspace->dQw, &dqw_sq_or_dw_dqw);
        a += gpu_workspace->lambda_max_Q * dqw_sq_or_dw_dqw;
        if (!gpu_workspace->empty_A) {
            gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->QATdy, &c_term);
            d_term = gpu_workspace->lambda_max_Q;
        }
    }

    a = a > 1e-12 ? a : 1e-12;
    b = b > 1e-12 ? b : 1e-12;

    if (gpu_workspace->Q_is_diag && !gpu_workspace->empty_A) {
        double left = 1e-12;
        double right = 1e12;
        double phi = (sqrt(5.0) - 1.0) / 2.0;
        double x1 = right - phi * (right - left);
        double x2 = left + phi * (right - left);
        double f1;
        double f2;
        size_t iter;

        gpu_sigma_diag_eval_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                            gpu_workspace->diag_Q.values,
                                                            gpu_workspace->QATdy.values,
                                                            x1,
                                                            gpu_workspace->n);
        gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->diag_tmp, &c_term);

        f1 = a * x1 + b / x1 + x1 * x1 * c_term;

        gpu_sigma_diag_eval_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                            gpu_workspace->diag_Q.values,
                                                            gpu_workspace->QATdy.values,
                                                            x2,
                                                            gpu_workspace->n);
        gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->diag_tmp, &c_term);
        f2 = a * x2 + b / x2 + x2 * x2 * c_term;

        for (iter = 0; iter < 200; ++iter) {
            if (fabs(right - left) <= 1e-12 * fmax(1.0, fabs(left))) {
                break;
            }
            if (f1 > f2) {
                left = x1;
                x1 = x2;
                f1 = f2;
                x2 = left + phi * (right - left);

                gpu_sigma_diag_eval_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                    gpu_workspace->diag_Q.values,
                                                                    gpu_workspace->QATdy.values,
                                                                    x2,
                                                                    gpu_workspace->n);
                gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->diag_tmp, &c_term);
                f2 = a * x2 + b / x2 + x2 * x2 * c_term;
            } else {
                right = x2;
                x2 = x1;
                f2 = f1;
                x1 = right - phi * (right - left);

                gpu_sigma_diag_eval_kernel<<<blocks_n, threads_n>>>(gpu_workspace->diag_tmp.values,
                                                                    gpu_workspace->diag_Q.values,
                                                                    gpu_workspace->QATdy.values,
                                                                    x1,
                                                                    gpu_workspace->n);
                gpu_vector_dot(blas_handle, &gpu_workspace->ATdy, &gpu_workspace->diag_tmp, &c_term);
                f1 = a * x1 + b / x1 + x1 * x1 * c_term;
            }
        }

        sigma_estimation = f1 < f2 ? x1 : x2;
    } else if (!gpu_workspace->Q_is_diag && !gpu_workspace->empty_A) {
        c_term = c_term > 0.0 ? c_term : 0.0;
        d_term = d_term > 0.0 ? d_term : 0.0;
        sigma_estimation = golden_search_sigma_local(a, b, c_term, d_term);
    } else {
        sigma_estimation = sqrt(b / a);
    }

    sigma_estimation = clamp_double_local(sigma_estimation, 1e-12, 1e12);
    fact = (restart_info->best_gap > 0.0 && isfinite(restart_info->best_gap))
               ? exp(-0.05 * (restart_info->current_gap / restart_info->best_gap))
               : 1.0;
    sigma_cand = exp(fact * log(sigma_estimation) +
                     (1.0 - fact) * log(restart_info->best_sigma > 0.0 ? restart_info->best_sigma : sigma_old));

    temp_1 = fmax(fmin(residuals->err_Rd_org_bar, residuals->err_Rp_org_bar),
                  fmin(residuals->rel_gap_bar, restart_info->current_gap));
    if (temp_1 <= 9e-10 && residuals->err_Rp_org_bar > 0.0) {
        ratio_infeas_org = residuals->err_Rd_org_bar / residuals->err_Rp_org_bar;
        if (temp_1 > 5e-10) {
            kappa = clamp_double_local(sqrt(ratio_infeas_org), 1e-2, 100.0);
        } else {
            kappa = clamp_double_local(ratio_infeas_org, 1e-2, 100.0);
        }
    }

    gpu_workspace->sigma = clamp_double_local(kappa * sigma_cand, 1e-12, 1e12);
    if (gpu_workspace->Q_is_diag && fabs(gpu_workspace->sigma - sigma_old) > 1e-15 && threads_n > 0U) {
        gpu_update_diag_factors_kernel<<<blocks_n, threads_n>>>(gpu_workspace->fact1.values,
                                                                gpu_workspace->fact2.values,
                                                                gpu_workspace->fact.values,
                                                                gpu_workspace->fact_M.values,
                                                                gpu_workspace->diag_Q.values,
                                                                gpu_workspace->sigma,
                                                                gpu_workspace->n);
    }

    *out_sigma = gpu_workspace->sigma;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_restart_apply_value(HPRQPGPUWorkspace *gpu_workspace,
                                    const HPRQPGPUProblem *gpu_problem)
{
    cudaMemcpy(gpu_workspace->x.values,
               gpu_workspace->x_bar.values,
               gpu_workspace->n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy(gpu_workspace->w.values,
               gpu_workspace->w_bar.values,
               gpu_workspace->n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy(gpu_workspace->last_x.values,
               gpu_workspace->x_bar.values,
               gpu_workspace->n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    cudaMemcpy(gpu_workspace->last_w.values,
               gpu_workspace->w_bar.values,
               gpu_workspace->n * sizeof(double),
               cudaMemcpyDeviceToDevice);

    if (!gpu_workspace->empty_A) {
        cudaMemcpy(gpu_workspace->y.values,
                   gpu_workspace->y_bar.values,
                   gpu_workspace->m * sizeof(double),
                   cudaMemcpyDeviceToDevice);

        cudaMemcpy(gpu_workspace->last_y.values,
                   gpu_workspace->y_bar.values,
                   gpu_workspace->m * sizeof(double),
                   cudaMemcpyDeviceToDevice);

        gpu_spmv_apply_at_y_bar(gpu_workspace, gpu_problem);

        cudaMemcpy(gpu_workspace->ATy.values,
                   gpu_workspace->ATy_bar.values,
                   gpu_workspace->n * sizeof(double),
                   cudaMemcpyDeviceToDevice);

        cudaMemcpy(gpu_workspace->last_ATy.values,
                   gpu_workspace->ATy_bar.values,
                   gpu_workspace->n * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    }

    if (gpu_workspace->noC) {
        cudaMemcpy(gpu_workspace->Qw.values,
                   gpu_workspace->Qw_bar.values,
                   gpu_workspace->n * sizeof(double),
                   cudaMemcpyDeviceToDevice);

        cudaMemcpy(gpu_workspace->last_Qw.values,
                   gpu_workspace->Qw_bar.values,
                   gpu_workspace->n * sizeof(double),
                   cudaMemcpyDeviceToDevice);
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_restart_apply_parallel_value(HPRQPGPUWorkspace *gpu_workspace)
{
    HPRQPStatus status;

    if (gpu_workspace == NULL || !gpu_workspace->allocated) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = map_cuda_status(cudaMemcpy(gpu_workspace->x.values,
                                        gpu_workspace->x_bar.values,
                                        gpu_workspace->n * sizeof(double),
                                        cudaMemcpyDeviceToDevice));
    if (status != HPRQP_STATUS_OK) return status;
    status = map_cuda_status(cudaMemcpy(gpu_workspace->w.values,
                                        gpu_workspace->w_bar.values,
                                        gpu_workspace->n * sizeof(double),
                                        cudaMemcpyDeviceToDevice));
    if (status != HPRQP_STATUS_OK) return status;
    status = map_cuda_status(cudaMemcpy(gpu_workspace->last_x.values,
                                        gpu_workspace->x_bar.values,
                                        gpu_workspace->n * sizeof(double),
                                        cudaMemcpyDeviceToDevice));
    if (status != HPRQP_STATUS_OK) return status;
    status = map_cuda_status(cudaMemcpy(gpu_workspace->last_w.values,
                                        gpu_workspace->w_bar.values,
                                        gpu_workspace->n * sizeof(double),
                                        cudaMemcpyDeviceToDevice));
    if (status != HPRQP_STATUS_OK) return status;
    status = map_cuda_status(cudaMemcpy(gpu_workspace->Qw.values,
                                        gpu_workspace->Qw_bar.values,
                                        gpu_workspace->n * sizeof(double),
                                        cudaMemcpyDeviceToDevice));
    if (status != HPRQP_STATUS_OK) return status;
    status = map_cuda_status(cudaMemcpy(gpu_workspace->last_Qw.values,
                                        gpu_workspace->Qw_bar.values,
                                        gpu_workspace->n * sizeof(double),
                                        cudaMemcpyDeviceToDevice));
    if (status != HPRQP_STATUS_OK) return status;

    if (!gpu_workspace->empty_A) {
        status = map_cuda_status(cudaMemcpy(gpu_workspace->y.values,
                                            gpu_workspace->y_bar.values,
                                            gpu_workspace->m * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(gpu_workspace->last_y.values,
                                            gpu_workspace->y_bar.values,
                                            gpu_workspace->m * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(gpu_workspace->ATy.values,
                                            gpu_workspace->ATy_bar.values,
                                            gpu_workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(gpu_workspace->last_ATy.values,
                                            gpu_workspace->ATy_bar.values,
                                            gpu_workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
    }

    if (gpu_workspace->p > 0U) {
        status = map_cuda_status(cudaMemcpy(
            gpu_workspace->qc_dual.values,
            gpu_workspace->qc_dual_bar.values,
            gpu_workspace->qc_cone_dim * sizeof(double),
            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(
            gpu_workspace->last_qc_dual.values,
            gpu_workspace->qc_dual_bar.values,
            gpu_workspace->qc_cone_dim * sizeof(double),
            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(
            gpu_workspace->BTs.values,
            gpu_workspace->BTs_bar.values,
            gpu_workspace->n * sizeof(double),
            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(
            gpu_workspace->last_BTs.values,
            gpu_workspace->BTs_bar.values,
            gpu_workspace->n * sizeof(double),
            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
    }

    return HPRQP_STATUS_OK;
}
