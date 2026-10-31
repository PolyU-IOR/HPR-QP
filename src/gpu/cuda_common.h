#ifndef HPRQP_CUDA_COMMON_H
#define HPRQP_CUDA_COMMON_H

#include "gpu/context.h"
#include "gpu/control.h"
#include "gpu/eigs.h"
#include "gpu/gpu.h"
#include "gpu/iteration.h"
#include "gpu/residuals.h"
#include "gpu/scale.h"
#include "gpu/spmv.h"
#include "gpu/workspace.h"
#include "cpu/scale.h"

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

static const double kSpmvAlpha = 1.0;
static const double kSpmvBeta = 0.0;
static const size_t kSpmvBenchmarkRepetitions = 30;

enum {
    HPRQP_GPU_BATCH_REDUCE_DOUBLE_COUNT = 16,
    HPRQP_GPU_BATCH_REDUCE_INDEX_COUNT = 8
};

static inline HPRQPStatus convert_host_indices_to_gpu(const size_t *host_values,
                                                      size_t count,
                                                      HPRQPGPUCsrIndex *gpu_values)
{
    size_t i;

    if (count == 0) {
        return HPRQP_STATUS_OK;
    }
    if (host_values == NULL || gpu_values == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < count; ++i) {
        if (host_values[i] > (size_t)INT32_MAX) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
        gpu_values[i] = (HPRQPGPUCsrIndex)host_values[i];
    }

    return HPRQP_STATUS_OK;
}

static inline HPRQPStatus map_cuda_status(cudaError_t cuda_status)
{
    return cuda_status == cudaSuccess ? HPRQP_STATUS_OK : HPRQP_STATUS_NOT_IMPLEMENTED;
}

static inline HPRQPStatus map_cusparse_status(cusparseStatus_t cusparse_status)
{
    switch (cusparse_status) {
    case CUSPARSE_STATUS_SUCCESS:
        return HPRQP_STATUS_OK;
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return HPRQP_STATUS_ALLOCATION_FAILED;
    case CUSPARSE_STATUS_INVALID_VALUE:
        return HPRQP_STATUS_INVALID_ARGUMENT;
    default:
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
}

static inline HPRQPStatus map_cublas_status(cublasStatus_t cublas_status)
{
    switch (cublas_status) {
    case CUBLAS_STATUS_SUCCESS:
        return HPRQP_STATUS_OK;
    case CUBLAS_STATUS_ALLOC_FAILED:
        return HPRQP_STATUS_ALLOCATION_FAILED;
    case CUBLAS_STATUS_INVALID_VALUE:
        return HPRQP_STATUS_INVALID_ARGUMENT;
    default:
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
}

static inline const char *spmv_mode_name_local(HPRQPSpmvMode mode)
{
    switch (mode) {
    case HPRQP_SPMV_MODE_AUTO:
        return "auto";
    case HPRQP_SPMV_MODE_CUSPARSE:
        return "cusparse";
    case HPRQP_SPMV_MODE_CUSTOMIZED:
        return "customized";
    default:
        return "unknown";
    }
}

static inline double clamp_double_local(double value, double lower, double upper)
{
    if (value < lower) {
        return lower;
    }
    if (value > upper) {
        return upper;
    }
    return value;
}

static inline double sigma_objective_local(double x, double a, double b, double c, double d)
{
    return a * x + b / x + c * x * x / (1.0 + d * x);
}

static inline double golden_search_sigma_local(double a, double b, double c, double d)
{
    double left = 1e-12;
    double right = 1e12;
    double inv_phi = (sqrt(5.0) - 1.0) / 2.0;
    double inv_phi_sq = (3.0 - sqrt(5.0)) / 2.0;
    double x1 = left + inv_phi_sq * (right - left);
    double x2 = left + inv_phi * (right - left);
    double f1 = sigma_objective_local(x1, a, b, c, d);
    double f2 = sigma_objective_local(x2, a, b, c, d);
    size_t iter;

    for (iter = 0; iter < 256; ++iter) {
        if (fabs(right - left) <= 1e-13 * fmax(1.0, fmax(fabs(left), fabs(right)))) {
            break;
        }
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + inv_phi * (right - left);
            f2 = sigma_objective_local(x2, a, b, c, d);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = left + inv_phi_sq * (right - left);
            f1 = sigma_objective_local(x1, a, b, c, d);
        }
    }

    return 0.5 * (left + right);
}

static inline cublasHandle_t gpu_blas_handle(const HPRQPGPUWorkspace *workspace)
{
    return workspace == NULL ? NULL : (cublasHandle_t)workspace->blas_handle;
}

static inline double *gpu_batch_reduce_values(HPRQPGPUWorkspace *workspace)
{
    return workspace == NULL ? NULL : (double *)workspace->reduce_buffer;
}

static inline int *gpu_batch_reduce_indices(HPRQPGPUWorkspace *workspace)
{
    unsigned char *base;

    if (workspace == NULL || workspace->reduce_buffer == NULL) {
        return NULL;
    }
    base = (unsigned char *)workspace->reduce_buffer;
    return (int *)(base + HPRQP_GPU_BATCH_REDUCE_DOUBLE_COUNT * sizeof(double));
}

static inline HPRQPStatus gpu_batch_reduce_begin(cublasHandle_t handle)
{
    return map_cublas_status(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
}

static inline HPRQPStatus gpu_batch_reduce_end(cublasHandle_t handle)
{
    return map_cublas_status(
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

static inline HPRQPStatus gpu_raw_dot_device(cublasHandle_t handle,
                                             size_t length,
                                             const double *lhs,
                                             const double *rhs,
                                             double *device_result)
{
    return map_cublas_status(
        cublasDdot(handle,
                   (int)length,
                   lhs,
                   1,
                   rhs,
                   1,
                   device_result));
}

static inline HPRQPStatus gpu_raw_iamax_device(cublasHandle_t handle,
                                               size_t length,
                                               const double *values,
                                               int *device_index)
{
    return map_cublas_status(
        cublasIdamax(handle,
                     (int)length,
                     values,
                     1,
                     device_index));
}

static inline HPRQPStatus gpu_raw_asum_device(cublasHandle_t handle,
                                              size_t length,
                                              const double *values,
                                              double *device_result)
{
    return map_cublas_status(
        cublasDasum(handle,
                    (int)length,
                    values,
                    1,
                    device_result));
}

__global__ static void gpu_batch_gather_abs_kernel(
    double *device_results,
    size_t result_slot,
    const int *device_indices,
    size_t index_slot,
    const double *values,
    size_t length)
{
    int index;

    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    index = device_indices[index_slot];
    device_results[result_slot] =
        index > 0 && (size_t)index <= length
            ? fabs(values[(size_t)index - 1])
            : 0.0;
}

static inline HPRQPStatus gpu_batch_gather_abs(
    double *device_results,
    size_t result_slot,
    const int *device_indices,
    size_t index_slot,
    const double *values,
    size_t length)
{
    gpu_batch_gather_abs_kernel<<<1, 1>>>(device_results,
                                          result_slot,
                                          device_indices,
                                          index_slot,
                                          values,
                                          length);
    return map_cuda_status(cudaGetLastError());
}

static inline HPRQPStatus gpu_batch_reduce_copy_to_host(
    const HPRQPGPUWorkspace *workspace,
    double *host_results,
    size_t count)
{
    if (workspace == NULL || host_results == NULL ||
        workspace->reduce_buffer == NULL ||
        count > HPRQP_GPU_BATCH_REDUCE_DOUBLE_COUNT) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    return map_cuda_status(cudaMemcpy(host_results,
                                      workspace->reduce_buffer,
                                      count * sizeof(double),
                                      cudaMemcpyDeviceToHost));
}

static inline HPRQPStatus gpu_raw_dot(cublasHandle_t handle,
                                      size_t length,
                                      const double *lhs,
                                      const double *rhs,
                                      double *result)
{
    cublasDdot(handle, (int)length, lhs, 1, rhs, 1, result);
    return HPRQP_STATUS_OK;
}

static inline HPRQPStatus gpu_vector_dot(cublasHandle_t handle,
                                         const HPRQPGPUVector *lhs,
                                         const HPRQPGPUVector *rhs,
                                         double *result)
{
    return gpu_raw_dot(handle, lhs->length, lhs->values, rhs->values, result);
}

static inline HPRQPStatus gpu_raw_norm2(cublasHandle_t handle,
                                        size_t length,
                                        const double *values,
                                        double *result)
{
    double norm_sq = 0.0;
    HPRQPStatus status;

    status = gpu_raw_dot(handle, length, values, values, &norm_sq);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    *result = sqrt(norm_sq > 0.0 ? norm_sq : 0.0);
    return HPRQP_STATUS_OK;
}

static inline HPRQPStatus gpu_raw_max_abs(cublasHandle_t handle,
                                          size_t length,
                                          const double *values,
                                          double *result)
{
    int index = 0;
    double value = 0.0;

    cublasIdamax(handle, (int)length, values, 1, &index);
    cudaMemcpy(&value, values + (size_t)(index - 1), sizeof(double), cudaMemcpyDeviceToHost);

    *result = fabs(value);
    return HPRQP_STATUS_OK;
}

#endif
