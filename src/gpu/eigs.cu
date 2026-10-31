#include "cuda_common.h"

#include "shared/parallel.h"
#include "shared/eigen_probe.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

__global__ static void gpu_fill_vector_kernel(double *values, double scalar, size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    values[i] = scalar;
}

__global__ static void gpu_scale_copy_kernel(double *dst,
                                             const double *src,
                                             double scale,
                                             size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    dst[i] = src[i] * scale;
}

__global__ static void gpu_residual_from_eigenpair_kernel(double *residual,
                                                          const double *z,
                                                          const double *q,
                                                          double lambda,
                                                          size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    residual[i] = z[i] - lambda * q[i];
}

__global__ static void gpu_init_probe_kernel(double *values, size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    values[i] = 1.0 + ((double)i + 1.0) / ((double)length + 1.0);
}

__global__ static void gpu_init_retry_probe_kernel(double *values, size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < length) {
        values[i] = hprqp_eigen_retry_probe_value(i);
    }
}

__global__ static void gpu_extract_diag_kernel(double *diag_q,
                                               double *offdiag_abs,
                                               const HPRQPGPUCsrIndex *row_ptr,
                                               const HPRQPGPUCsrIndex *col_ind,
                                               const double *values,
                                               size_t n)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double diag_value = 0.0;
    double offdiag_value = 0.0;
    size_t idx;

    if (row >= n) {
        return;
    }

    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        size_t col = (size_t)col_ind[idx];
        double value = values[idx];

        if (col == row) {
            diag_value += value;
        } else {
            double abs_value = fabs(value);
            if (abs_value > offdiag_value) {
                offdiag_value = abs_value;
            }
        }
    }

    diag_q[row] = diag_value;
    offdiag_abs[row] = offdiag_value;
}

static HPRQPStatus gpu_launch_fill_vector(HPRQPGPUVector *vector, double scalar, unsigned int threads)
{
    unsigned int blocks;

    if (vector == NULL || (vector->length > 0 && vector->values == NULL)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (vector->length == 0) {
        return HPRQP_STATUS_OK;
    }

    blocks = (unsigned int)((vector->length + threads - 1) / threads);
    gpu_fill_vector_kernel<<<blocks, threads>>>(vector->values, scalar, vector->length);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_launch_scale_copy(double *dst,
                                         const double *src,
                                         double scale,
                                         size_t length,
                                         unsigned int threads)
{
    unsigned int blocks;

    if (length == 0) {
        return HPRQP_STATUS_OK;
    }

    blocks = (unsigned int)((length + threads - 1) / threads);
    gpu_scale_copy_kernel<<<blocks, threads>>>(dst, src, scale, length);
    return map_cuda_status(cudaGetLastError());
}

static HPRQPStatus gpu_launch_probe(double *values, size_t length, unsigned int threads)
{
    unsigned int blocks;

    if (length == 0) {
        return HPRQP_STATUS_OK;
    }

    blocks = (unsigned int)((length + threads - 1) / threads);
    gpu_init_probe_kernel<<<blocks, threads>>>(values, length);
    return map_cuda_status(cudaGetLastError());
}

static HPRQPStatus gpu_extract_diag_info_local(cublasHandle_t handle,
                                               HPRQPGPUWorkspace *workspace,
                                               const HPRQPGPUProblem *model,
                                               int *q_is_diag,
                                               double *lambda_max_q)
{
    HPRQPStatus status;
    double max_offdiag = 0.0;
    double max_diag = 0.0;

    if (q_is_diag == NULL || lambda_max_q == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    gpu_extract_diag_kernel<<<workspace->blocks_n, workspace->threads_n>>>(workspace->diag_Q.values,
                                                                            workspace->reduce_tmp1.values,
                                                                            model->Q.row_ptr,
                                                                            model->Q.col_ind,
                                                                            model->Q.values,
                                                                            workspace->n);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    status = gpu_raw_max_abs(handle, workspace->n, workspace->reduce_tmp1.values, &max_offdiag);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    *q_is_diag = max_offdiag <= 1e-15 ? 1 : 0;
    if (*q_is_diag) {
        status = gpu_raw_max_abs(handle, workspace->n, workspace->diag_Q.values, &max_diag);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        *lambda_max_q = max_diag;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_power_iteration_q_local(HPRQPGPUWorkspace *workspace,
                                               const HPRQPGPUProblem *model,
                                               double *lambda_max)
{
    cublasHandle_t handle = gpu_blas_handle(workspace);
    HPRQPStatus status;
    size_t iteration;
    double *q = workspace->x_bar.values;
    double *z = workspace->Qx.values;

    if (lambda_max == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->n == 0 || model->Q.nnz == 0) {
        *lambda_max = 0.0;
        return HPRQP_STATUS_OK;
    }

    if (!workspace->spmv_Q.prepared) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_launch_probe(z, workspace->n, workspace->threads_n);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    *lambda_max = 1.0;
    for (iteration = 0; iteration < 5000; ++iteration) {
        double q_norm;
        double z_norm;
        double error_norm;

        status = gpu_raw_norm2(handle, workspace->n, z, &q_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        if (q_norm <= 1e-16) {
            *lambda_max = 0.0;
            break;
        }

        status = gpu_launch_scale_copy(q, z, 1.0 / q_norm, workspace->n, workspace->threads_n);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_spmv_apply_q_x_bar(workspace, model);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_dot(handle, workspace->n, q, z, lambda_max);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_norm2(handle, workspace->n, z, &z_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }

        gpu_residual_from_eigenpair_kernel<<<workspace->blocks_n, workspace->threads_n>>>(q,
                                                                                            z,
                                                                                            q,
                                                                                            *lambda_max,
                                                                                            workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_norm2(handle, workspace->n, q, &error_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }

        if (error_norm / (z_norm + fabs(*lambda_max) + 1e-16) < 1e-4) {
            break;
        }
    }

    if (*lambda_max < 0.0) {
        *lambda_max = 0.0;
    }

    status = gpu_launch_fill_vector(&workspace->x_bar, 0.0, workspace->threads_n);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_launch_fill_vector(&workspace->Qx, 0.0, workspace->threads_n);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_power_iteration_a_local(HPRQPGPUWorkspace *workspace,
                                               const HPRQPGPUProblem *model,
                                               double *lambda_max)
{
    cublasHandle_t handle = gpu_blas_handle(workspace);
    HPRQPStatus status;
    size_t iteration;
    double *q = workspace->y_bar.values;
    double *z = workspace->Ax.values;

    if (lambda_max == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->m == 0 || workspace->empty_A) {
        *lambda_max = 0.0;
        return HPRQP_STATUS_OK;
    }

    if (!workspace->spmv_AT.prepared || !workspace->spmv_A.prepared) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_launch_probe(z, workspace->m, workspace->threads_m);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    *lambda_max = 1.0;
    for (iteration = 0; iteration < 5000; ++iteration) {
        double q_norm;
        double z_norm;
        double error_norm;

        status = gpu_raw_norm2(handle, workspace->m, z, &q_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        if (q_norm <= 1e-16) {
            *lambda_max = 0.0;
            break;
        }

        status = gpu_launch_scale_copy(q, z, 1.0 / q_norm, workspace->m, workspace->threads_m);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_spmv_apply_at_y_bar(workspace, model);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = map_cuda_status(cudaMemcpy(workspace->tempv.values,
                                            workspace->ATy_bar.values,
                                            workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_spmv_apply_a_tempv(workspace, model);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_dot(handle, workspace->m, q, z, lambda_max);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_norm2(handle, workspace->m, z, &z_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }

        gpu_residual_from_eigenpair_kernel<<<workspace->blocks_m, workspace->threads_m>>>(q,
                                                                                            z,
                                                                                            q,
                                                                                            *lambda_max,
                                                                                            workspace->m);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_norm2(handle, workspace->m, q, &error_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }

        if (error_norm / (z_norm + fabs(*lambda_max) + 1e-16) < 1e-4) {
            break;
        }
    }

    if (*lambda_max < 0.0) {
        *lambda_max = 0.0;
    }

    status = gpu_launch_fill_vector(&workspace->y_bar, 0.0, workspace->threads_m);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_launch_fill_vector(&workspace->tempv, 0.0, workspace->threads_n);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_launch_fill_vector(&workspace->ATy_bar, 0.0, workspace->threads_n);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_launch_fill_vector(&workspace->Ax, 0.0, workspace->threads_m);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_power_iteration_aqka_local(HPRQPGPUWorkspace *workspace,
                                                  const HPRQPGPUProblem *model,
                                                  int q_power,
                                                  int *retry_moments,
                                                  double *lambda_max)
{
    cublasHandle_t handle = gpu_blas_handle(workspace);
    HPRQPStatus status;
    size_t iteration;
    bool retry_probe = *retry_moments != 0;
    double *q = workspace->y_bar.values;
    double *z = workspace->Ax.values;

    if (lambda_max == NULL || q_power < 1 || q_power > 2) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->m == 0 || workspace->empty_A || model->Q.nnz == 0) {
        *lambda_max = 0.0;
        return HPRQP_STATUS_OK;
    }
    if (!workspace->spmv_AT.prepared || !workspace->spmv_A.prepared ||
        !workspace->spmv_Q.prepared) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (retry_probe) {
        gpu_init_retry_probe_kernel<<<workspace->blocks_m,
                                       workspace->threads_m>>>(z, workspace->m);
        status = map_cuda_status(cudaGetLastError());
    } else {
        status = gpu_launch_probe(z, workspace->m, workspace->threads_m);
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    *lambda_max = 1.0;
    for (iteration = 0; iteration < 5000; ++iteration) {
        double q_norm;
        double z_norm;
        double error_norm;

        status = gpu_raw_norm2(handle, workspace->m, z, &q_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        if (q_norm == 0.0 || (!retry_probe && q_norm <= 1e-16)) {
            if (!retry_probe && q_power == 1) {
                gpu_init_retry_probe_kernel<<<workspace->blocks_m,
                                               workspace->threads_m>>>(z, workspace->m);
                status = map_cuda_status(cudaGetLastError());
                if (status != HPRQP_STATUS_OK) return status;
                retry_probe = true;
                *retry_moments = 1;
                iteration = (size_t)-1;
                continue;
            }
            *lambda_max = 0.0;
            break;
        }
        status = gpu_launch_scale_copy(q,
                                       z,
                                       1.0 / q_norm,
                                       workspace->m,
                                       workspace->threads_m);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_spmv_apply_at_y_bar(workspace, model);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = map_cuda_status(cudaMemcpy(workspace->ATdy.values,
                                            workspace->ATy_bar.values,
                                            workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_spmv_apply_q_atdy(workspace, model);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        if (q_power == 2) {
            status = map_cuda_status(cudaMemcpy(workspace->ATdy.values,
                                                workspace->QATdy.values,
                                                workspace->n * sizeof(double),
                                                cudaMemcpyDeviceToDevice));
            if (status != HPRQP_STATUS_OK) return status;
            status = gpu_spmv_apply_q_atdy(workspace, model);
            if (status != HPRQP_STATUS_OK) return status;
        }
        status = map_cuda_status(cudaMemcpy(workspace->tempv.values,
                                            workspace->QATdy.values,
                                            workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_spmv_apply_a_tempv(workspace, model);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_dot(handle, workspace->m, q, z, lambda_max);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_norm2(handle, workspace->m, z, &z_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        gpu_residual_from_eigenpair_kernel<<<workspace->blocks_m,
                                             workspace->threads_m>>>(q,
                                                                    z,
                                                                    q,
                                                                    *lambda_max,
                                                                    workspace->m);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        status = gpu_raw_norm2(handle, workspace->m, q, &error_norm);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        {
            const double relative_scale = z_norm + fabs(*lambda_max);
            const double error = retry_probe
                ? (z_norm > 0.0 ? error_norm / relative_scale : 0.0)
                : error_norm / (relative_scale + 1e-16);
            if (!retry_probe && q_power == 1 && error < 1e-4 &&
                (z_norm == 0.0 || error_norm / relative_scale >= 1e-4)) {
                gpu_init_retry_probe_kernel<<<workspace->blocks_m,
                                               workspace->threads_m>>>(z, workspace->m);
                status = map_cuda_status(cudaGetLastError());
                if (status != HPRQP_STATUS_OK) return status;
                retry_probe = true;
                *retry_moments = 1;
                iteration = (size_t)-1;
                continue;
            }
            if (error < 1e-4) break;
        }
    }

    if (*lambda_max < 0.0) {
        *lambda_max = 0.0;
    }
    status = gpu_launch_fill_vector(&workspace->y_bar, 0.0, workspace->threads_m);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_launch_fill_vector(&workspace->Ax, 0.0, workspace->threads_m);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_launch_fill_vector(&workspace->ATy_bar, 0.0, workspace->threads_n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_launch_fill_vector(&workspace->ATdy, 0.0, workspace->threads_n);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_launch_fill_vector(&workspace->QATdy, 0.0, workspace->threads_n);
    if (status != HPRQP_STATUS_OK) return status;
    return gpu_launch_fill_vector(&workspace->tempv, 0.0, workspace->threads_n);
}

HPRQPStatus gpu_estimate_eigenvalues(HPRQPGPUWorkspace *workspace,
                                     const HPRQPGPUProblem *model,
                                     const HPRQPParams *params)
{
    cublasHandle_t handle;
    HPRQPStatus status;
    const bool profile_setup = std::getenv("HPRQP_PROFILE_SETUP") != nullptr;
    const auto profile_start =
        profile_setup ? std::chrono::steady_clock::now()
                      : std::chrono::steady_clock::time_point{};
    auto profile_last = profile_start;
    double diag_seconds = 0.0;
    double power_q_seconds = 0.0;
    double power_a_seconds = 0.0;
    double power_aqa_seconds = 0.0;
    double power_aq2a_seconds = 0.0;
    bool moment_selector_requested;
    int retry_moments = 0;

    if (workspace == NULL || model == NULL || params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    handle = gpu_blas_handle(workspace);
    if (handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    workspace->method = params->method;
    workspace->parallel_majorizer_mode = params->parallel_majorizer_mode;
    moment_selector_requested =
        workspace->parallel_majorizer_mode ==
        HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR;

    status = gpu_extract_diag_info_local(handle,
                                         workspace,
                                         model,
                                         &workspace->Q_is_diag,
                                         &workspace->lambda_max_Q);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        diag_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (!workspace->Q_is_diag) {
        status = gpu_power_iteration_q_local(workspace, model, &workspace->lambda_max_Q);
        if (profile_setup) {
            const auto now = std::chrono::steady_clock::now();
            power_q_seconds = std::chrono::duration<double>(now - profile_last).count();
            profile_last = now;
        }
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        workspace->lambda_max_Q *= params->eig_factor;
    }

    status = gpu_power_iteration_a_local(workspace, model, &workspace->lambda_max_A);
    if (profile_setup) {
        const auto now = std::chrono::steady_clock::now();
        power_a_seconds = std::chrono::duration<double>(now - profile_last).count();
        profile_last = now;
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    workspace->lambda_max_A *= params->eig_factor;

    if (workspace->method == HPRQP_METHOD_PARALLEL) {
        if (workspace->parallel_majorizer_mode ==
                HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER ||
            workspace->parallel_majorizer_mode ==
                HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT ||
            moment_selector_requested) {
            if (!workspace->empty_A && workspace->lambda_max_A > 0.0 &&
                workspace->lambda_max_Q > 0.0) {
                status = gpu_power_iteration_aqka_local(
                    workspace, model, 1, &retry_moments, &workspace->lambda_max_AQA);
                if (profile_setup) {
                    const auto now = std::chrono::steady_clock::now();
                    power_aqa_seconds =
                        std::chrono::duration<double>(now - profile_last).count();
                    profile_last = now;
                }
                if (status != HPRQP_STATUS_OK) {
                    return status;
                }
                workspace->lambda_max_AQA *= params->eig_factor;
                if (workspace->lambda_max_AQA >
                    workspace->lambda_max_A * workspace->lambda_max_Q) {
                    workspace->lambda_max_AQA =
                        workspace->lambda_max_A * workspace->lambda_max_Q;
                }
            } else {
                workspace->lambda_max_AQA = 0.0;
            }
            workspace->zeta_AQ = parallel_normalized_coupling(
                workspace->lambda_max_AQA,
                workspace->lambda_max_A,
                workspace->lambda_max_Q);
            if (workspace->parallel_majorizer_mode ==
                    HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT ||
                moment_selector_requested) {
                if (workspace->lambda_max_AQA > 0.0) {
                    status = gpu_power_iteration_aqka_local(
                        workspace, model, 2, &retry_moments, &workspace->lambda_max_AQ2A);
                    if (profile_setup) {
                        const auto now = std::chrono::steady_clock::now();
                        power_aq2a_seconds =
                            std::chrono::duration<double>(now - profile_last).count();
                        profile_last = now;
                    }
                    if (status != HPRQP_STATUS_OK) return status;
                    workspace->lambda_max_AQ2A *= params->eig_factor;
                    if (workspace->lambda_max_AQ2A >
                        workspace->lambda_max_Q * workspace->lambda_max_AQA) {
                        workspace->lambda_max_AQ2A =
                            workspace->lambda_max_Q * workspace->lambda_max_AQA;
                    }
                } else {
                    workspace->lambda_max_AQ2A = 0.0;
                }
            } else {
                workspace->lambda_max_AQ2A = NAN;
            }
            if (moment_selector_requested) {
                double delta_2;
                workspace->parallel_majorizer_mode =
                    parallel_select_moment_majorizer(
                        workspace->lambda_max_Q,
                        workspace->lambda_max_AQA,
                        workspace->lambda_max_AQ2A,
                        &delta_2);
                if (params->verbose) {
                    std::fprintf(stderr,
                                 "HPRQP_MOMENT_SELECTOR delta_2=%.17g "
                                 "threshold=%.17g selected=%s "
                                 "lambda_max_Q=%.17g lambda_max_AQA=%.17g "
                                 "lambda_max_AQ2A=%.17g\n",
                                 delta_2,
                                 HPRQP_SECOND_MOMENT_DELTA_TOLERANCE,
                                 workspace->parallel_majorizer_mode ==
                                         HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER
                                     ? "one_moment" : "two_moment",
                                 workspace->lambda_max_Q,
                                 workspace->lambda_max_AQA,
                                 workspace->lambda_max_AQ2A);
                    std::fflush(stderr);
                }
            }
        } else {
            workspace->lambda_max_AQA =
                workspace->lambda_max_A * workspace->lambda_max_Q;
            workspace->lambda_max_AQ2A = NAN;
            workspace->zeta_AQ = 1.0;
        }
    } else {
        workspace->lambda_max_AQA = NAN;
        workspace->lambda_max_AQ2A = NAN;
        workspace->zeta_AQ = NAN;
    }
    workspace->eigenvalues_initialized = 1;

    if (profile_setup) {
        const auto profile_done = std::chrono::steady_clock::now();
        std::fprintf(stderr,
                     "HPRQP_PROFILE eigs: diag=%.6f s, power_Q=%.6f s, "
                     "power_A=%.6f s, power_AQA=%.6f s, "
                     "power_AQ2A=%.6f s, tail=%.6f s, "
                     "total=%.6f s\n",
                     diag_seconds,
                     power_q_seconds,
                     power_a_seconds,
                     power_aqa_seconds,
                     power_aq2a_seconds,
                     std::chrono::duration<double>(profile_done - profile_last).count(),
                     std::chrono::duration<double>(profile_done - profile_start).count());
    }

    return HPRQP_STATUS_OK;
}
