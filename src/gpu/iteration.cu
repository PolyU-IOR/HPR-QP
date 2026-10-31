#include "cuda_common.h"

#include "gpu/qcqp.h"

#include <cooperative_groups.h>
#include <cfloat>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace cg = cooperative_groups;

static HPRQPStatus gpu_vector_copy_device_to_device(HPRQPGPUVector *dst, const HPRQPGPUVector *src)
{
    if (dst == NULL || src == NULL || dst->length != src->length) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (dst->length == 0) {
        return HPRQP_STATUS_OK;
    }

    return map_cuda_status(cudaMemcpy(dst->values,
                                      src->values,
                                      dst->length * sizeof(double),
                                      cudaMemcpyDeviceToDevice));
}

static HPRQPStatus gpu_workspace_copy_q_benchmark_state(HPRQPGPUWorkspace *dst,
                                                        const HPRQPGPUWorkspace *src)
{
    HPRQPStatus status;

    if (dst == NULL || src == NULL || dst->m != src->m || dst->n != src->n) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_vector_copy_device_to_device(&dst->z_bar, &src->z_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->x, &src->x);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->x_hat, &src->x_hat);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->x_bar, &src->x_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->w, &src->w);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->w_bar, &src->w_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->dx, &src->dx);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->Qw, &src->Qw);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->Qw_bar, &src->Qw_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->tempv, &src->tempv);
    if (status != HPRQP_STATUS_OK) return status;

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_workspace_copy_a_benchmark_state(HPRQPGPUWorkspace *dst,
                                                        const HPRQPGPUWorkspace *src)
{
    HPRQPStatus status;

    if (dst == NULL || src == NULL || dst->m != src->m || dst->n != src->n) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_vector_copy_device_to_device(&dst->y, &src->y);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->y_bar, &src->y_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->dy, &src->dy);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->s, &src->s);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->w, &src->w);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->w_bar, &src->w_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->dw, &src->dw);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->Ax, &src->Ax);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->ATy, &src->ATy);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->ATy_bar, &src->ATy_bar);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_vector_copy_device_to_device(&dst->ATdy, &src->ATdy);
    if (status != HPRQP_STATUS_OK) return status;

    return HPRQP_STATUS_OK;
}

static __device__ __forceinline__ double interval_project_device(double value, double lower, double upper)
{
    return value < lower ? lower : (value > upper ? upper : value);
}

static __device__ __forceinline__ double csr_row_dot_device(const HPRQPGPUCsrIndex *row_ptr,
                                                            const HPRQPGPUCsrIndex *col_ind,
                                                            const double *values,
                                                            const double *vector,
                                                            size_t row)
{
    size_t idx;
    size_t row_start = (size_t)row_ptr[row];
    size_t row_end = (size_t)row_ptr[row + 1];
    double sum = 0.0;

    for (idx = row_start; idx < row_end; ++idx) {
        sum += values[idx] * vector[(size_t)col_ind[idx]];
    }

    return sum;
}

__global__ static void gpu_parallel_update_xw_kernel(
    double *dx,
    double *dw,
    double *x_bar,
    double *x_hat,
    double *z_bar,
    double *w_bar,
    double *x,
    double *w,
    const double *last_x,
    const double *last_w,
    const double *Qw,
    const double *ATy,
    const double *c,
    const double *l,
    const double *u,
    double sigma,
    double w_gamma,
    double w_inv,
    double halpern_fact1,
    double halpern_fact2,
    int write_metric,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    double x_i;
    double w_i;
    double z_raw;
    double x_bar_i;
    double x_hat_i;
    double w_bar_i;

    if (i >= n) {
        return;
    }
    x_i = x[i];
    w_i = w[i];
    z_raw = x_i + sigma * (-Qw[i] + ATy[i] - c[i]);
    x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    x_hat_i = 2.0 * x_bar_i - x_i;
    w_bar_i = (w_gamma * w_i + x_hat_i) * w_inv;

    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    w_bar[i] = w_bar_i;
    if (write_metric) {
        dx[i] = x_bar_i - x_i;
        dw[i] = w_bar_i - w_i;
    }
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w[i] = halpern_fact2 * (2.0 * w_bar_i - w_i) +
           halpern_fact1 * last_w[i];
}

__global__ static void gpu_parallel_update_xw_fused_at_row_kernel(
    double *dx,
    double *dw,
    double *x_bar,
    double *x_hat,
    double *z_bar,
    double *w_bar,
    double *x,
    double *w,
    const double *last_x,
    const double *last_w,
    const double *Qw,
    double *ATy,
    const double *y,
    const HPRQPGPUCsrIndex *at_row_ptr,
    const HPRQPGPUCsrIndex *at_col_ind,
    const double *at_values,
    const double *c,
    const double *l,
    const double *u,
    double sigma,
    double w_gamma,
    double w_inv,
    double halpern_fact1,
    double halpern_fact2,
    int write_metric,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    double x_i;
    double w_i;
    double z_raw;
    double x_bar_i;
    double x_hat_i;
    double w_bar_i;
    double aty_i;

    if (i >= n) {
        return;
    }
    x_i = x[i];
    w_i = w[i];
    aty_i = csr_row_dot_device(at_row_ptr,
                               at_col_ind,
                               at_values,
                               y,
                               i);
    ATy[i] = aty_i;
    z_raw = x_i + sigma * (-Qw[i] + aty_i - c[i]);
    x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    x_hat_i = 2.0 * x_bar_i - x_i;
    w_bar_i = (w_gamma * w_i + x_hat_i) * w_inv;

    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    w_bar[i] = w_bar_i;
    if (write_metric) {
        dx[i] = x_bar_i - x_i;
        dw[i] = w_bar_i - w_i;
    }
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w[i] = halpern_fact2 * (2.0 * w_bar_i - w_i) +
           halpern_fact1 * last_w[i];
}

__global__ static void gpu_parallel_update_xw_fused_q_row_kernel(
    double *dx,
    double *dw,
    double *x_bar,
    double *x_hat,
    double *z_bar,
    double *w_bar,
    double *x,
    double *w,
    const double *last_x,
    const double *last_w,
    double *Qw,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    double *ATy,
    const double *y,
    const HPRQPGPUCsrIndex *at_row_ptr,
    const HPRQPGPUCsrIndex *at_col_ind,
    const double *at_values,
    const double *c,
    const double *l,
    const double *u,
    double sigma,
    double w_gamma,
    double w_inv,
    double halpern_fact1,
    double halpern_fact2,
    int fuse_at,
    int write_metric,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    double x_i;
    double w_i;
    double qw_i;
    double aty_i;
    double z_raw;
    double x_bar_i;
    double x_hat_i;
    double w_bar_i;

    if (i >= n) {
        return;
    }
    x_i = x[i];
    w_i = w[i];
    qw_i = csr_row_dot_device(q_row_ptr,
                              q_col_ind,
                              q_values,
                              w,
                              i);
    Qw[i] = qw_i;
    if (fuse_at) {
        aty_i = csr_row_dot_device(at_row_ptr,
                                   at_col_ind,
                                   at_values,
                                   y,
                                   i);
        ATy[i] = aty_i;
    } else {
        aty_i = ATy[i];
    }
    z_raw = x_i + sigma * (-qw_i + aty_i - c[i]);
    x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    x_hat_i = 2.0 * x_bar_i - x_i;
    w_bar_i = (w_gamma * w_i + x_hat_i) * w_inv;

    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    w_bar[i] = w_bar_i;
    if (write_metric) {
        dx[i] = x_bar_i - x_i;
        dw[i] = w_bar_i - w_i;
    }
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w[i] = halpern_fact2 * (2.0 * w_bar_i - w_i) +
           halpern_fact1 * last_w[i];
}

__global__ static void gpu_parallel_update_y_kernel(
    double *dy,
    double *s,
    double *y_bar,
    double *y,
    const double *last_y,
    const double *Ax,
    const double *AL,
    const double *AU,
    double y_rho,
    double y_inv,
    double halpern_fact1,
    double halpern_fact2,
    int write_metric,
    size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    double y_i;
    double s_raw;
    double s_proj;
    double y_bar_i;

    if (i >= m) {
        return;
    }
    y_i = y[i];
    s_raw = Ax[i] - y_rho * y_i;
    s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    y_bar_i = (s_proj - s_raw) * y_inv;
    s[i] = s_proj;
    y_bar[i] = y_bar_i;
    if (write_metric) {
        dy[i] = y_bar_i - y_i;
    }
    y[i] = halpern_fact2 * (2.0 * y_bar_i - y_i) +
           halpern_fact1 * last_y[i];
}

__global__ static void gpu_parallel_update_y_fused_a_row_kernel(
    double *dy,
    double *s,
    double *y_bar,
    double *y,
    const double *last_y,
    double *Ax,
    const double *x_hat,
    const HPRQPGPUCsrIndex *a_row_ptr,
    const HPRQPGPUCsrIndex *a_col_ind,
    const double *a_values,
    const double *AL,
    const double *AU,
    double y_rho,
    double y_inv,
    double halpern_fact1,
    double halpern_fact2,
    int write_metric,
    size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    double y_i;
    double s_raw;
    double s_proj;
    double y_bar_i;
    double ax_i;

    if (i >= m) {
        return;
    }
    y_i = y[i];
    ax_i = csr_row_dot_device(a_row_ptr,
                              a_col_ind,
                              a_values,
                              x_hat,
                              i);
    Ax[i] = ax_i;
    s_raw = ax_i - y_rho * y_i;
    s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    y_bar_i = (s_proj - s_raw) * y_inv;
    s[i] = s_proj;
    y_bar[i] = y_bar_i;
    if (write_metric) {
        dy[i] = y_bar_i - y_i;
    }
    y[i] = halpern_fact2 * (2.0 * y_bar_i - y_i) +
           halpern_fact1 * last_y[i];
}

typedef struct HPRQPGPUParallelPersistentArgs {
    size_t n;
    size_t m;
    size_t q_nonempty_count;
    int has_constraints;
    const HPRQPGPUCsrIndex *q_nonempty_rows;
    const HPRQPGPUCsrIndex *q_row_ptr;
    const HPRQPGPUCsrIndex *q_col_ind;
    const uint16_t *q_col16;
    const HPRQPGPUCsrIndex *q_col16_base;
    const double *q_values;
    const HPRQPGPUCsrIndex *a_row_ptr;
    const HPRQPGPUCsrIndex *a_col_ind;
    const uint16_t *a_col16;
    const HPRQPGPUCsrIndex *a_col16_base;
    const double *a_values;
    const HPRQPGPUCsrIndex *at_row_ptr;
    const HPRQPGPUCsrIndex *at_col_ind;
    const uint16_t *at_col16;
    const HPRQPGPUCsrIndex *at_col16_base;
    const double *at_values;
    const double *c;
    const double *l;
    const double *u;
    const double *AL;
    const double *AU;
    const double *last_x;
    const double *last_w;
    const double *last_y;
    const double *halpern_fact1;
    const double *halpern_fact2;
    double *x;
    double *w;
    double *y;
    double *scratch_x;
    double *scratch_w;
    double *scratch_y;
    double *x_bar;
    double *x_hat;
    double *z_bar;
    double *w_bar;
    double *y_bar;
    double *s;
    double *Qw;
    double *ATy;
    double *Ax;
    double *dx;
    double *dw;
    double *dy;
    double sigma;
    double w_gamma;
    double w_inv;
    double y_rho;
    double y_inv;
    int write_metric;
    int use_col16;
    int x_lower_positive_zero;
    int constraints_positive_zero_equality;
    unsigned int q_group_width;
} HPRQPGPUParallelPersistentArgs;

template <unsigned int FixedRowLength>
static __device__ __forceinline__ double persistent_csr_row_dot_impl(
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *col_ind,
    const uint16_t *col16,
    const HPRQPGPUCsrIndex *col16_base,
    const double *values,
    const double *vector,
    size_t row,
    int use_col16)
{
    const size_t row_start = (size_t)row_ptr[row];
    const size_t row_end = (size_t)row_ptr[row + 1U];
    double sum = 0.0;

    if (use_col16 == 1) {
        if constexpr (FixedRowLength == 6U) {
            if (row_end - row_start == 6U) {
                sum += values[row_start] *
                       vector[(size_t)col16[row_start]];
                sum += values[row_start + 1U] *
                       vector[(size_t)col16[row_start + 1U]];
                sum += values[row_start + 2U] *
                       vector[(size_t)col16[row_start + 2U]];
                sum += values[row_start + 3U] *
                       vector[(size_t)col16[row_start + 3U]];
                sum += values[row_start + 4U] *
                       vector[(size_t)col16[row_start + 4U]];
                sum += values[row_start + 5U] *
                       vector[(size_t)col16[row_start + 5U]];
                return sum;
            }
        }
        for (size_t index = row_start; index < row_end; ++index) {
            sum += values[index] * vector[(size_t)col16[index]];
        }
    } else if (use_col16 == 2 && col16_base[row] >= 0) {
        const size_t base = (size_t)col16_base[row];
        if constexpr (FixedRowLength == 6U) {
            if (row_end - row_start == 6U) {
                sum += values[row_start] *
                       vector[base + (size_t)col16[row_start]];
                sum += values[row_start + 1U] *
                       vector[base + (size_t)col16[row_start + 1U]];
                sum += values[row_start + 2U] *
                       vector[base + (size_t)col16[row_start + 2U]];
                sum += values[row_start + 3U] *
                       vector[base + (size_t)col16[row_start + 3U]];
                sum += values[row_start + 4U] *
                       vector[base + (size_t)col16[row_start + 4U]];
                sum += values[row_start + 5U] *
                       vector[base + (size_t)col16[row_start + 5U]];
                return sum;
            }
        }
        if constexpr (FixedRowLength == 10U) {
            if (row_end - row_start == 10U) {
                sum += values[row_start] *
                       vector[base + (size_t)col16[row_start]];
                sum += values[row_start + 1U] *
                       vector[base + (size_t)col16[row_start + 1U]];
                sum += values[row_start + 2U] *
                       vector[base + (size_t)col16[row_start + 2U]];
                sum += values[row_start + 3U] *
                       vector[base + (size_t)col16[row_start + 3U]];
                sum += values[row_start + 4U] *
                       vector[base + (size_t)col16[row_start + 4U]];
                sum += values[row_start + 5U] *
                       vector[base + (size_t)col16[row_start + 5U]];
                sum += values[row_start + 6U] *
                       vector[base + (size_t)col16[row_start + 6U]];
                sum += values[row_start + 7U] *
                       vector[base + (size_t)col16[row_start + 7U]];
                sum += values[row_start + 8U] *
                       vector[base + (size_t)col16[row_start + 8U]];
                sum += values[row_start + 9U] *
                       vector[base + (size_t)col16[row_start + 9U]];
                return sum;
            }
        }
        for (size_t index = row_start; index < row_end; ++index) {
            sum += values[index] *
                   vector[base + (size_t)col16[index]];
        }
    } else {
        if constexpr (FixedRowLength == 6U) {
            if (row_end - row_start == 6U) {
                sum += values[row_start] *
                       vector[(size_t)col_ind[row_start]];
                sum += values[row_start + 1U] *
                       vector[(size_t)col_ind[row_start + 1U]];
                sum += values[row_start + 2U] *
                       vector[(size_t)col_ind[row_start + 2U]];
                sum += values[row_start + 3U] *
                       vector[(size_t)col_ind[row_start + 3U]];
                sum += values[row_start + 4U] *
                       vector[(size_t)col_ind[row_start + 4U]];
                sum += values[row_start + 5U] *
                       vector[(size_t)col_ind[row_start + 5U]];
                return sum;
            }
        }
        for (size_t index = row_start; index < row_end; ++index) {
            sum += values[index] * vector[(size_t)col_ind[index]];
        }
    }
    return sum;
}

static __device__ __forceinline__ double persistent_csr_row_dot(
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *col_ind,
    const uint16_t *col16,
    const HPRQPGPUCsrIndex *col16_base,
    const double *values,
    const double *vector,
    size_t row,
    int use_col16)
{
    return persistent_csr_row_dot_impl<0U>(row_ptr,
                                           col_ind,
                                           col16,
                                           col16_base,
                                           values,
                                           vector,
                                           row,
                                           use_col16);
}

static __device__ __forceinline__ double persistent_csr_warp_dot(
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *col_ind,
    const double *values,
    const double *vector,
    size_t row,
    unsigned int lane)
{
    const size_t row_start = (size_t)row_ptr[row];
    const size_t row_end = (size_t)row_ptr[row + 1];
    double sum = 0.0;

    for (size_t idx = row_start + (size_t)lane;
         idx < row_end;
         idx += 32U) {
        sum += values[idx] * vector[(size_t)col_ind[idx]];
    }
    sum += __shfl_down_sync(0xffffffffU, sum, 16);
    sum += __shfl_down_sync(0xffffffffU, sum, 8);
    sum += __shfl_down_sync(0xffffffffU, sum, 4);
    sum += __shfl_down_sync(0xffffffffU, sum, 2);
    sum += __shfl_down_sync(0xffffffffU, sum, 1);
    return sum;
}

static __device__ __forceinline__ double persistent_csr_warp_dot16(
    const HPRQPGPUCsrIndex *row_ptr,
    const uint16_t *col_ind,
    const double *values,
    const double *vector,
    size_t row,
    unsigned int lane)
{
    const size_t row_start = (size_t)row_ptr[row];
    const size_t row_end = (size_t)row_ptr[row + 1];
    double sum = 0.0;

    for (size_t idx = row_start + (size_t)lane;
         idx < row_end;
         idx += 32U) {
        sum += values[idx] * vector[(size_t)col_ind[idx]];
    }
    sum += __shfl_down_sync(0xffffffffU, sum, 16);
    sum += __shfl_down_sync(0xffffffffU, sum, 8);
    sum += __shfl_down_sync(0xffffffffU, sum, 4);
    sum += __shfl_down_sync(0xffffffffU, sum, 2);
    sum += __shfl_down_sync(0xffffffffU, sum, 1);
    return sum;
}

static __device__ __forceinline__ void persistent_q_group_spmv(
    const HPRQPGPUParallelPersistentArgs &args,
    const double *vector,
    double *warp_sums,
    size_t q_slots,
    unsigned int lane,
    size_t warp_in_block,
    size_t warps_per_block)
{
    const unsigned int group_width = args.q_group_width;
    const size_t warps_per_group = (size_t)group_width >> 5U;
    const size_t groups_per_block =
        warps_per_block / warps_per_group;
    const size_t group_in_block = warp_in_block / warps_per_group;
    const size_t warp_in_group = warp_in_block % warps_per_group;
    const size_t group_lane = warp_in_group * 32U + (size_t)lane;
    const size_t rounds =
        (q_slots + groups_per_block - 1U) / groups_per_block;

    for (size_t round = 0U; round < rounds; ++round) {
        const size_t slot =
            group_in_block + round * groups_per_block;
        const size_t task =
            (size_t)blockIdx.x + slot * (size_t)gridDim.x;
        const int active =
            slot < q_slots && task < args.q_nonempty_count;
        size_t row = 0U;
        double sum = 0.0;

        if (active) {
            row = (size_t)args.q_nonempty_rows[task];
            const size_t row_start = (size_t)args.q_row_ptr[row];
            const size_t row_end = (size_t)args.q_row_ptr[row + 1U];
            for (size_t idx = row_start + group_lane;
                 idx < row_end;
                 idx += (size_t)group_width) {
                const size_t column =
                    args.use_col16
                        ? (size_t)args.q_col16[idx]
                        : (size_t)args.q_col_ind[idx];
                sum += args.q_values[idx] * vector[column];
            }
        }
        sum += __shfl_down_sync(0xffffffffU, sum, 16);
        sum += __shfl_down_sync(0xffffffffU, sum, 8);
        sum += __shfl_down_sync(0xffffffffU, sum, 4);
        sum += __shfl_down_sync(0xffffffffU, sum, 2);
        sum += __shfl_down_sync(0xffffffffU, sum, 1);
        if (lane == 0U) {
            warp_sums[warp_in_block] = sum;
        }
        __syncthreads();

        if (warp_in_group == 0U) {
            double group_sum =
                (size_t)lane < warps_per_group
                    ? warp_sums[group_in_block * warps_per_group +
                                (size_t)lane]
                    : 0.0;
            group_sum +=
                __shfl_down_sync(0xffffffffU, group_sum, 16);
            group_sum +=
                __shfl_down_sync(0xffffffffU, group_sum, 8);
            group_sum +=
                __shfl_down_sync(0xffffffffU, group_sum, 4);
            group_sum +=
                __shfl_down_sync(0xffffffffU, group_sum, 2);
            group_sum +=
                __shfl_down_sync(0xffffffffU, group_sum, 1);
            if (lane == 0U && active) {
                args.Qw[row] = group_sum;
            }
        }
        __syncthreads();
    }
}

__global__ static void gpu_parallel_persistent_kernel(
    HPRQPGPUParallelPersistentArgs args,
    size_t batch_count)
{
    cg::grid_group grid = cg::this_grid();
    __shared__ double q_warp_sums[16];
    const size_t global_thread =
        (size_t)blockIdx.x * (size_t)blockDim.x +
        (size_t)threadIdx.x;
    const size_t thread_count =
        (size_t)gridDim.x * (size_t)blockDim.x;
    const unsigned int lane = threadIdx.x & 31U;
    const size_t warp_in_block = (size_t)threadIdx.x >> 5;
    const size_t warps_per_block = (size_t)blockDim.x >> 5;
    const size_t q_slots =
        (args.q_nonempty_count + (size_t)gridDim.x - 1U) /
        (size_t)gridDim.x;
    const size_t n_slots =
        (args.n + (size_t)gridDim.x - 1U) /
        (size_t)gridDim.x;
    const size_t a_slots =
        args.has_constraints
            ? (args.m + (size_t)gridDim.x - 1U) /
                  (size_t)gridDim.x
            : 0U;
    double *current_x = args.x;
    double *current_w = args.w;
    double *current_y = args.y;
    double *next_x = args.scratch_x;
    double *next_w = args.scratch_w;
    double *next_y = args.scratch_y;

    for (size_t row = global_thread; row < args.n; row += thread_count) {
        args.Qw[row] = 0.0;
    }
    if (args.q_group_width == 32U) {
        for (size_t slot = warp_in_block;
             slot < q_slots;
             slot += warps_per_block) {
            const size_t task =
                (size_t)blockIdx.x + slot * (size_t)gridDim.x;
            if (task < args.q_nonempty_count) {
                const size_t row = (size_t)args.q_nonempty_rows[task];
                const double qw =
                    args.use_col16
                        ? persistent_csr_warp_dot16(args.q_row_ptr,
                                                    args.q_col16,
                                                    args.q_values,
                                                    current_w,
                                                    row,
                                                    lane)
                        : persistent_csr_warp_dot(args.q_row_ptr,
                                                  args.q_col_ind,
                                                  args.q_values,
                                                  current_w,
                                                  row,
                                                  lane);
                if (lane == 0U) {
                    args.Qw[row] = qw;
                }
            }
        }
    } else {
        persistent_q_group_spmv(args,
                                current_w,
                                q_warp_sums,
                                q_slots,
                                lane,
                                warp_in_block,
                                warps_per_block);
    }
    grid.sync();

    for (size_t batch_iter = 0; batch_iter < batch_count; ++batch_iter) {
        const double alpha = args.halpern_fact1[batch_iter];
        const int final_iteration = batch_iter + 1U == batch_count;
        const int write_metric =
            args.write_metric && final_iteration;

        for (size_t slot = warp_in_block;
             slot < n_slots;
             slot += warps_per_block) {
            const size_t row =
                (size_t)blockIdx.x + slot * (size_t)gridDim.x;
            double aty = 0.0;

            if (row < args.n && args.has_constraints) {
                aty = args.use_col16
                          ? persistent_csr_warp_dot16(args.at_row_ptr,
                                                      args.at_col16,
                                                      args.at_values,
                                                      current_y,
                                                      row,
                                                      lane)
                          : persistent_csr_warp_dot(args.at_row_ptr,
                                                    args.at_col_ind,
                                                    args.at_values,
                                                    current_y,
                                                    row,
                                                    lane);
            }
            if (lane == 0U && row < args.n) {
                const double qw = args.Qw[row];
                const double x_i = current_x[row];
                const double w_i = current_w[row];
                const double z_raw =
                    x_i + args.sigma * (-qw + aty - args.c[row]);
                const double x_bar_i =
                    interval_project_device(z_raw,
                                            args.l[row],
                                            args.u[row]);
                const double x_hat_i = 2.0 * x_bar_i - x_i;
                const double w_delta_i =
                    args.w_inv * (x_hat_i - w_i);
                const double w_bar_i = w_i + w_delta_i;
                const double w_reflected_i = w_i + 2.0 * w_delta_i;

                args.x_hat[row] = x_hat_i;
                if (final_iteration) {
                    args.ATy[row] = aty;
                    args.x_bar[row] = x_bar_i;
                    args.z_bar[row] = (x_bar_i - z_raw) / args.sigma;
                    args.w_bar[row] = w_bar_i;
                }
                if (write_metric) {
                    args.dx[row] = x_bar_i - x_i;
                    args.dw[row] = w_bar_i - w_i;
                }
                next_x[row] =
                    x_hat_i + alpha * (args.last_x[row] - x_hat_i);
                next_w[row] =
                    w_reflected_i +
                    alpha * (args.last_w[row] - w_reflected_i);
            }
        }

        grid.sync();

        if (args.q_group_width == 32U) {
            const size_t active_q_slots = final_iteration ? 0U : q_slots;
            const size_t phase_slots = active_q_slots + a_slots;

            for (size_t slot = warp_in_block;
                 slot < phase_slots;
                 slot += warps_per_block) {
                if (slot < active_q_slots) {
                    const size_t task =
                        (size_t)blockIdx.x +
                        slot * (size_t)gridDim.x;
                    if (task < args.q_nonempty_count) {
                        const size_t row =
                            (size_t)args.q_nonempty_rows[task];
                        const double qw =
                            args.use_col16
                                ? persistent_csr_warp_dot16(
                                      args.q_row_ptr,
                                      args.q_col16,
                                      args.q_values,
                                      next_w,
                                      row,
                                      lane)
                                : persistent_csr_warp_dot(
                                      args.q_row_ptr,
                                      args.q_col_ind,
                                      args.q_values,
                                      next_w,
                                      row,
                                      lane);
                        if (lane == 0U) {
                            args.Qw[row] = qw;
                        }
                    }
                    continue;
                }

                if (args.has_constraints) {
                    const size_t a_slot = slot - active_q_slots;
                    const size_t row =
                        (size_t)blockIdx.x +
                        a_slot * (size_t)gridDim.x;
                    if (row >= args.m) {
                        continue;
                    }
                    const double ax =
                        args.use_col16
                            ? persistent_csr_warp_dot16(
                                  args.a_row_ptr,
                                  args.a_col16,
                                  args.a_values,
                                  args.x_hat,
                                  row,
                                  lane)
                            : persistent_csr_warp_dot(
                                  args.a_row_ptr,
                                  args.a_col_ind,
                                  args.a_values,
                                  args.x_hat,
                                  row,
                                  lane);
                    if (lane == 0U) {
                        const double y_i = current_y[row];
                        const double s_raw = ax - args.y_rho * y_i;
                        const double s_proj =
                            interval_project_device(s_raw,
                                                    args.AL[row],
                                                    args.AU[row]);
                        const double y_delta_i =
                            args.y_inv * (s_proj - ax);
                        const double y_bar_i = y_i + y_delta_i;
                        const double y_reflected_i =
                            y_i + 2.0 * y_delta_i;

                        if (final_iteration) {
                            args.Ax[row] = ax;
                            args.s[row] = s_proj;
                            args.y_bar[row] = y_bar_i;
                        }
                        if (write_metric) {
                            args.dy[row] = y_bar_i - y_i;
                        }
                        next_y[row] =
                            y_reflected_i +
                            alpha *
                                (args.last_y[row] - y_reflected_i);
                    }
                }
            }
        } else {
            if (!final_iteration) {
                persistent_q_group_spmv(args,
                                        next_w,
                                        q_warp_sums,
                                        q_slots,
                                        lane,
                                        warp_in_block,
                                        warps_per_block);
            }
            if (args.has_constraints) {
                for (size_t a_slot = warp_in_block;
                     a_slot < a_slots;
                     a_slot += warps_per_block) {
                    const size_t row =
                        (size_t)blockIdx.x +
                        a_slot * (size_t)gridDim.x;
                    if (row >= args.m) {
                        continue;
                    }
                    const double ax =
                        args.use_col16
                            ? persistent_csr_warp_dot16(
                                  args.a_row_ptr,
                                  args.a_col16,
                                  args.a_values,
                                  args.x_hat,
                                  row,
                                  lane)
                            : persistent_csr_warp_dot(
                                  args.a_row_ptr,
                                  args.a_col_ind,
                                  args.a_values,
                                  args.x_hat,
                                  row,
                                  lane);
                    if (lane == 0U) {
                        const double y_i = current_y[row];
                        const double s_raw = ax - args.y_rho * y_i;
                        const double s_proj =
                            interval_project_device(s_raw,
                                                    args.AL[row],
                                                    args.AU[row]);
                        const double y_delta_i =
                            args.y_inv * (s_proj - ax);
                        const double y_bar_i = y_i + y_delta_i;
                        const double y_reflected_i =
                            y_i + 2.0 * y_delta_i;

                        if (final_iteration) {
                            args.Ax[row] = ax;
                            args.s[row] = s_proj;
                            args.y_bar[row] = y_bar_i;
                        }
                        if (write_metric) {
                            args.dy[row] = y_bar_i - y_i;
                        }
                        next_y[row] =
                            y_reflected_i +
                            alpha *
                                (args.last_y[row] - y_reflected_i);
                    }
                }
            }
        }

        grid.sync();

        {
            double *swap_value;
            swap_value = current_x;
            current_x = next_x;
            next_x = swap_value;
            swap_value = current_w;
            current_w = next_w;
            next_w = swap_value;
            if (args.has_constraints) {
                swap_value = current_y;
                current_y = next_y;
                next_y = swap_value;
            }
        }
    }

    if ((batch_count & 1U) != 0U) {
        for (size_t i = global_thread; i < args.n; i += thread_count) {
            args.x[i] = current_x[i];
            args.w[i] = current_w[i];
        }
        if (args.has_constraints) {
            for (size_t i = global_thread; i < args.m; i += thread_count) {
                args.y[i] = current_y[i];
            }
        }
        grid.sync();
    }
}

/*
 * Race-free batched form of the ordinary row kernels.  Every CSR row is
 * accumulated in the same left-to-right order as csr_row_dot_device.  Between
 * unchecked iterations current_x stores the preceding x_hat instead of the
 * already blended x iterate: the next iteration evaluates the identical
 * Halpern blend immediately before using x.  This lets the same buffer serve
 * as both the A*x_hat input and the next x state, removing one full-vector
 * store without changing the scalar expression.  The final iteration writes
 * the ordinary materialized x state so all code outside this kernel observes
 * the original workspace representation.
 */
template <unsigned int FixedRowLength,
          int FixedColMode = -1,
          bool ExactSingleDiagonalQ = false>
__global__ static void gpu_parallel_persistent_row_kernel(
    HPRQPGPUParallelPersistentArgs args,
    size_t batch_count)
{
    cg::grid_group grid = cg::this_grid();
    const size_t global_thread =
        (size_t)blockIdx.x * (size_t)blockDim.x +
        (size_t)threadIdx.x;
    const size_t thread_count =
        (size_t)gridDim.x * (size_t)blockDim.x;
    double *current_x = args.x;
    double *current_w = args.w;
    double *current_y = args.y;
    double *next_x = args.scratch_x;
    double *next_w = args.scratch_w;
    double *next_y = args.scratch_y;
    const int use_col16 =
        FixedColMode >= 0 ? FixedColMode : args.use_col16;
    double previous_halpern_fact1 = 0.0;
    double previous_halpern_fact2 = 0.0;

    for (size_t batch_iter = 0U;
         batch_iter < batch_count;
         ++batch_iter) {
        const double halpern_fact1 = args.halpern_fact1[batch_iter];
        const double halpern_fact2 = args.halpern_fact2[batch_iter];
        const int final_iteration = batch_iter + 1U == batch_count;
        const int write_metric = args.write_metric && final_iteration;

        for (size_t row = global_thread;
             row < args.n;
             row += thread_count) {
            const double stored_x_i = current_x[row];
            double last_x_i = 0.0;
            double x_i;

            if (batch_iter != 0U || final_iteration) {
                last_x_i = args.last_x[row];
            }
            x_i = batch_iter == 0U
                      ? stored_x_i
                      : previous_halpern_fact2 * stored_x_i +
                            previous_halpern_fact1 * last_x_i;
            const double w_i = current_w[row];
            double qw_i;
            if constexpr (ExactSingleDiagonalQ) {
                const size_t q_begin = (size_t)args.q_row_ptr[row];
                const size_t q_end = (size_t)args.q_row_ptr[row + 1U];

                qw_i = 0.0;
                if (q_end > q_begin) {
                    qw_i += args.q_values[q_begin] * w_i;
                }
            } else {
                qw_i = persistent_csr_row_dot_impl<FixedRowLength>(
                    args.q_row_ptr,
                    args.q_col_ind,
                    args.q_col16,
                    args.q_col16_base,
                    args.q_values,
                    current_w,
                    row,
                    use_col16);
            }
            const double aty_i =
                args.has_constraints
                    ? persistent_csr_row_dot_impl<FixedRowLength>(
                          args.at_row_ptr,
                          args.at_col_ind,
                          args.at_col16,
                          args.at_col16_base,
                          args.at_values,
                          current_y,
                          row,
                          use_col16)
                    : 0.0;
            const double z_raw =
                x_i + args.sigma * (-qw_i + aty_i - args.c[row]);
            const double x_bar_i =
                args.x_lower_positive_zero
                    ? interval_project_device(z_raw,
                                              0.0,
                                              args.u[row])
                    : interval_project_device(z_raw,
                                              args.l[row],
                                              args.u[row]);
            const double x_hat_i = 2.0 * x_bar_i - x_i;
            const double w_bar_i =
                (args.w_gamma * w_i + x_hat_i) * args.w_inv;

            if (final_iteration) {
                args.x_hat[row] = x_hat_i;
                args.Qw[row] = qw_i;
                args.ATy[row] = aty_i;
                args.x_bar[row] = x_bar_i;
                args.z_bar[row] = (x_bar_i - z_raw) / args.sigma;
                args.w_bar[row] = w_bar_i;
            }
            if (write_metric) {
                args.dx[row] = x_bar_i - x_i;
                args.dw[row] = w_bar_i - w_i;
            }
            next_x[row] = final_iteration
                              ? halpern_fact2 * x_hat_i +
                                    halpern_fact1 * last_x_i
                              : x_hat_i;
            next_w[row] =
                halpern_fact2 * (2.0 * w_bar_i - w_i) +
                halpern_fact1 * args.last_w[row];
        }

        grid.sync();

        if (args.has_constraints) {
            for (size_t row = global_thread;
                 row < args.m;
                 row += thread_count) {
                const double y_i = current_y[row];
                const double ax_i =
                    persistent_csr_row_dot_impl<FixedRowLength>(
                        args.a_row_ptr,
                        args.a_col_ind,
                        args.a_col16,
                        args.a_col16_base,
                        args.a_values,
                        final_iteration ? args.x_hat : next_x,
                        row,
                        use_col16);
                const double s_raw = ax_i - args.y_rho * y_i;
                const double s_proj =
                    args.constraints_positive_zero_equality
                        ? interval_project_device(s_raw, 0.0, 0.0)
                        : interval_project_device(s_raw,
                                                  args.AL[row],
                                                  args.AU[row]);
                const double y_bar_i =
                    (s_proj - s_raw) * args.y_inv;

                if (final_iteration) {
                    args.Ax[row] = ax_i;
                    args.s[row] = s_proj;
                    args.y_bar[row] = y_bar_i;
                }
                if (write_metric) {
                    args.dy[row] = y_bar_i - y_i;
                }
                next_y[row] =
                    halpern_fact2 * (2.0 * y_bar_i - y_i) +
                    halpern_fact1 * args.last_y[row];
            }
        }

        grid.sync();

        {
            double *swap_value = current_x;
            current_x = next_x;
            next_x = swap_value;
            swap_value = current_w;
            current_w = next_w;
            next_w = swap_value;
            if (args.has_constraints) {
                swap_value = current_y;
                current_y = next_y;
                next_y = swap_value;
            }
        }
        previous_halpern_fact1 = halpern_fact1;
        previous_halpern_fact2 = halpern_fact2;
    }

    if ((batch_count & 1U) != 0U) {
        for (size_t row = global_thread;
             row < args.n;
             row += thread_count) {
            args.x[row] = current_x[row];
            args.w[row] = current_w[row];
        }
        if (args.has_constraints) {
            for (size_t row = global_thread;
                 row < args.m;
                 row += thread_count) {
                args.y[row] = current_y[row];
            }
        }
        grid.sync();
    }
}

/*
 * Pipeline Q*w for the next iteration beside the independent A*x_hat/y phase.
 * CSR rows and scalar expressions remain identical to the ordinary kernels.
 */
__device__ __forceinline__ static size_t
persistent_balanced_block_begin(size_t row_count)
{
    const size_t block_count = (size_t)gridDim.x;
    const size_t block_index = (size_t)blockIdx.x;
    const size_t rows_per_block = row_count / block_count;
    const size_t extra_rows = row_count % block_count;

    return block_index * rows_per_block +
           (block_index < extra_rows ? block_index : extra_rows);
}

__device__ __forceinline__ static size_t
persistent_balanced_block_end(size_t row_count)
{
    const size_t block_index = (size_t)blockIdx.x;
    const size_t begin = persistent_balanced_block_begin(row_count);
    const size_t rows_per_block = row_count / (size_t)gridDim.x;
    const size_t extra_rows = row_count % (size_t)gridDim.x;

    return begin + rows_per_block + (block_index < extra_rows ? 1U : 0U);
}

__global__ static void gpu_parallel_persistent_row_pipeline_kernel(
    HPRQPGPUParallelPersistentArgs args,
    size_t batch_count)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ double q_cache[];
    const size_t n_begin = persistent_balanced_block_begin(args.n);
    const size_t n_end = persistent_balanced_block_end(args.n);
    const size_t m_begin = persistent_balanced_block_begin(args.m);
    const size_t m_end = persistent_balanced_block_end(args.m);
    const size_t thread = (size_t)threadIdx.x;
    const size_t q_threads = (size_t)args.q_group_width;
    const size_t a_threads = (size_t)blockDim.x - q_threads;
    double *current_x = args.x;
    double *current_w = args.w;
    double *current_y = args.y;
    double *next_x = args.scratch_x;
    double *next_w = args.scratch_w;
    double *next_y = args.scratch_y;

    if (thread < q_threads) {
        for (size_t row = n_begin + thread;
             row < n_end;
             row += q_threads) {
            q_cache[row - n_begin] =
                persistent_csr_row_dot(args.q_row_ptr,
                                       args.q_col_ind,
                                       args.q_col16,
                                       args.q_col16_base,
                                       args.q_values,
                                       current_w,
                                       row,
                                       args.use_col16);
        }
    }
    grid.sync();

    for (size_t batch_iter = 0U;
         batch_iter < batch_count;
         ++batch_iter) {
        const double halpern_fact1 = args.halpern_fact1[batch_iter];
        const double halpern_fact2 = args.halpern_fact2[batch_iter];
        const int final_iteration = batch_iter + 1U == batch_count;
        const int write_metric = args.write_metric && final_iteration;

        for (size_t row = n_begin + thread;
             row < n_end;
             row += (size_t)blockDim.x) {
            const double x_i = current_x[row];
            const double w_i = current_w[row];
            const double qw_i = q_cache[row - n_begin];
            const double aty_i =
                args.has_constraints
                    ? persistent_csr_row_dot(args.at_row_ptr,
                                             args.at_col_ind,
                                             args.at_col16,
                                             args.at_col16_base,
                                             args.at_values,
                                             current_y,
                                             row,
                                             args.use_col16)
                    : 0.0;
            const double z_raw =
                x_i + args.sigma * (-qw_i + aty_i - args.c[row]);
            const double x_bar_i =
                args.x_lower_positive_zero
                    ? interval_project_device(z_raw,
                                              0.0,
                                              args.u[row])
                    : interval_project_device(z_raw,
                                              args.l[row],
                                              args.u[row]);
            const double x_hat_i = 2.0 * x_bar_i - x_i;
            const double w_bar_i =
                (args.w_gamma * w_i + x_hat_i) * args.w_inv;

            args.x_hat[row] = x_hat_i;
            if (final_iteration) {
                args.Qw[row] = qw_i;
                args.ATy[row] = aty_i;
                args.x_bar[row] = x_bar_i;
                args.z_bar[row] = (x_bar_i - z_raw) / args.sigma;
                args.w_bar[row] = w_bar_i;
            }
            if (write_metric) {
                args.dx[row] = x_bar_i - x_i;
                args.dw[row] = w_bar_i - w_i;
            }
            next_x[row] =
                halpern_fact2 * x_hat_i +
                halpern_fact1 * args.last_x[row];
            next_w[row] =
                halpern_fact2 * (2.0 * w_bar_i - w_i) +
                halpern_fact1 * args.last_w[row];
        }

        grid.sync();

        if (!final_iteration && thread < q_threads) {
            for (size_t row = n_begin + thread;
                 row < n_end;
                 row += q_threads) {
                q_cache[row - n_begin] =
                    persistent_csr_row_dot(args.q_row_ptr,
                                           args.q_col_ind,
                                           args.q_col16,
                                           args.q_col16_base,
                                           args.q_values,
                                           next_w,
                                           row,
                                           args.use_col16);
            }
        }

        if (args.has_constraints && thread >= q_threads) {
            const size_t a_thread = thread - q_threads;

            for (size_t row = m_begin + a_thread;
                 row < m_end;
                 row += a_threads) {
                const double y_i = current_y[row];
                const double ax_i =
                    persistent_csr_row_dot(args.a_row_ptr,
                                           args.a_col_ind,
                                           args.a_col16,
                                           args.a_col16_base,
                                           args.a_values,
                                           args.x_hat,
                                           row,
                                           args.use_col16);
                const double s_raw = ax_i - args.y_rho * y_i;
                const double s_proj =
                    args.constraints_positive_zero_equality
                        ? interval_project_device(s_raw, 0.0, 0.0)
                        : interval_project_device(s_raw,
                                                  args.AL[row],
                                                  args.AU[row]);
                const double y_bar_i =
                    (s_proj - s_raw) * args.y_inv;

                if (final_iteration) {
                    args.Ax[row] = ax_i;
                    args.s[row] = s_proj;
                    args.y_bar[row] = y_bar_i;
                }
                if (write_metric) {
                    args.dy[row] = y_bar_i - y_i;
                }
                next_y[row] =
                    halpern_fact2 * (2.0 * y_bar_i - y_i) +
                    halpern_fact1 * args.last_y[row];
            }
        }

        grid.sync();

        {
            double *swap_value = current_x;
            current_x = next_x;
            next_x = swap_value;
            swap_value = current_w;
            current_w = next_w;
            next_w = swap_value;
            if (args.has_constraints) {
                swap_value = current_y;
                current_y = next_y;
                next_y = swap_value;
            }
        }
    }

    if ((batch_count & 1U) != 0U) {
        for (size_t row = n_begin + thread;
             row < n_end;
             row += (size_t)blockDim.x) {
            args.x[row] = current_x[row];
            args.w[row] = current_w[row];
        }
        if (args.has_constraints) {
            for (size_t row = m_begin + thread;
                 row < m_end;
                 row += (size_t)blockDim.x) {
                args.y[row] = current_y[row];
            }
        }
        grid.sync();
    }
}

__global__ static void gpu_parallel_image_diff_kernel(
    double *dQw,
    double *ATdy,
    const double *Qw_bar,
    const double *Qw,
    const double *ATy_bar,
    const double *ATy,
    int has_constraints,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    dQw[i] = Qw_bar[i] - Qw[i];
    ATdy[i] = has_constraints ? ATy_bar[i] - ATy[i] : 0.0;
}

static HPRQPStatus gpu_parallel_finish_check(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem)
{
    HPRQPStatus status;

    if (problem->Q.nnz > 0) {
        status = gpu_spmv_apply_q_w_bar(workspace, problem);
    } else {
        status = map_cuda_status(cudaMemset(workspace->Qw_bar.values,
                                            0,
                                            workspace->n * sizeof(double)));
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (!workspace->empty_A) {
        status = gpu_spmv_apply_at_y_bar(workspace, problem);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    } else {
        /*
         * The persistent batch reuses ATy_bar as its alternating w buffer.
         * With no constraint rows there is no A^T*y_bar SpMV to overwrite
         * that scratch state, but the residual code still requires the
         * mathematical value A^T*y_bar = 0.
         */
        status = map_cuda_status(cudaMemset(workspace->ATy_bar.values,
                                            0,
                                            workspace->n * sizeof(double)));
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }
    gpu_parallel_image_diff_kernel<<<workspace->blocks_n,
                                     workspace->threads_n>>>(
        workspace->dQw.values,
        workspace->ATdy.values,
        workspace->Qw_bar.values,
        workspace->Qw.values,
        workspace->ATy_bar.values,
        workspace->ATy.values,
        workspace->empty_A ? 0 : 1,
        workspace->n);
    return map_cuda_status(cudaGetLastError());
}

static HPRQPStatus gpu_parallel_prepare_q_nonempty_rows(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem)
{
    HPRQPGPUCsrIndex *host_row_ptr = NULL;
    HPRQPGPUCsrIndex *host_columns = NULL;
    HPRQPGPUCsrIndex *host_rows = NULL;
    HPRQPGPUCsrIndex *device_rows = NULL;
    size_t count = 0U;
    int exact_single_diagonal = 1;
    HPRQPStatus status;

    if (workspace == NULL || problem == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->parallel_q_nonempty_rows != NULL) {
        status = map_cuda_status(cudaFree(
            workspace->parallel_q_nonempty_rows));
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        workspace->parallel_q_nonempty_rows = NULL;
        workspace->parallel_q_nonempty_count = 0U;
    }
    workspace->parallel_q_exact_single_diagonal = 0;
    if (problem->Q.nnz == 0U) {
        return HPRQP_STATUS_OK;
    }

    host_row_ptr = (HPRQPGPUCsrIndex *)std::malloc(
        (problem->Q.nrow + 1U) * sizeof(HPRQPGPUCsrIndex));
    host_columns = (HPRQPGPUCsrIndex *)std::malloc(
        problem->Q.nnz * sizeof(HPRQPGPUCsrIndex));
    if (host_row_ptr == NULL || host_columns == NULL) {
        std::free(host_columns);
        std::free(host_row_ptr);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    status = map_cuda_status(cudaMemcpy(
        host_row_ptr,
        problem->Q.row_ptr,
        (problem->Q.nrow + 1U) * sizeof(HPRQPGPUCsrIndex),
        cudaMemcpyDeviceToHost));
    if (status != HPRQP_STATUS_OK) {
        std::free(host_columns);
        std::free(host_row_ptr);
        return status;
    }
    status = map_cuda_status(cudaMemcpy(
        host_columns,
        problem->Q.col_ind,
        problem->Q.nnz * sizeof(HPRQPGPUCsrIndex),
        cudaMemcpyDeviceToHost));
    if (status != HPRQP_STATUS_OK) {
        std::free(host_columns);
        std::free(host_row_ptr);
        return status;
    }
    /*
     * This is deliberately stricter than Q_is_diag: preserving the CSR
     * trajectory requires zero or one stored entry per row, exactly on the
     * diagonal.  Duplicate diagonal entries and tiny off-diagonals fall back.
     */
    for (size_t row = 0U; row < problem->Q.nrow; ++row) {
        const HPRQPGPUCsrIndex begin = host_row_ptr[row];
        const HPRQPGPUCsrIndex end = host_row_ptr[row + 1U];

        if (begin < 0 || end < begin || (size_t)end > problem->Q.nnz) {
            std::free(host_columns);
            std::free(host_row_ptr);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        if (end > begin) {
            ++count;
        }
        if (end - begin > 1 ||
            (end - begin == 1 && host_columns[begin] !=
                                     (HPRQPGPUCsrIndex)row)) {
            exact_single_diagonal = 0;
        }
    }
    host_rows = (HPRQPGPUCsrIndex *)std::malloc(
        count * sizeof(HPRQPGPUCsrIndex));
    if (host_rows == NULL) {
        std::free(host_columns);
        std::free(host_row_ptr);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    count = 0U;
    for (size_t row = 0U; row < problem->Q.nrow; ++row) {
        if (host_row_ptr[row + 1U] > host_row_ptr[row]) {
            host_rows[count++] = (HPRQPGPUCsrIndex)row;
        }
    }
    std::free(host_columns);
    std::free(host_row_ptr);

    status = map_cuda_status(cudaMalloc((void **)&device_rows,
                                        count * sizeof(HPRQPGPUCsrIndex)));
    if (status != HPRQP_STATUS_OK) {
        std::free(host_rows);
        return status;
    }
    status = map_cuda_status(cudaMemcpy(
        device_rows,
        host_rows,
        count * sizeof(HPRQPGPUCsrIndex),
        cudaMemcpyHostToDevice));
    std::free(host_rows);
    if (status != HPRQP_STATUS_OK) {
        cudaFree(device_rows);
        return status;
    }
    workspace->parallel_q_nonempty_rows = device_rows;
    workspace->parallel_q_nonempty_count = count;
    workspace->parallel_q_exact_single_diagonal = exact_single_diagonal;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_parallel_convert_columns_to_u16(
    const HPRQPGPUMatrixCSR *matrix,
    uint16_t **device_columns)
{
    HPRQPGPUCsrIndex *host_columns = NULL;
    uint16_t *host_columns16 = NULL;
    uint16_t *columns16 = NULL;
    HPRQPStatus status;

    if (matrix == NULL || device_columns == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *device_columns = NULL;
    if (matrix->nnz == 0U) {
        return HPRQP_STATUS_OK;
    }
    host_columns = (HPRQPGPUCsrIndex *)std::malloc(
        matrix->nnz * sizeof(HPRQPGPUCsrIndex));
    host_columns16 = (uint16_t *)std::malloc(
        matrix->nnz * sizeof(uint16_t));
    if (host_columns == NULL || host_columns16 == NULL) {
        std::free(host_columns16);
        std::free(host_columns);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    status = map_cuda_status(cudaMemcpy(host_columns,
                                        matrix->col_ind,
                                        matrix->nnz *
                                            sizeof(HPRQPGPUCsrIndex),
                                        cudaMemcpyDeviceToHost));
    if (status != HPRQP_STATUS_OK) {
        std::free(host_columns16);
        std::free(host_columns);
        return status;
    }
    for (size_t index = 0U; index < matrix->nnz; ++index) {
        if (host_columns[index] < 0 ||
            (uint32_t)host_columns[index] >= 65536U) {
            std::free(host_columns16);
            std::free(host_columns);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        host_columns16[index] = (uint16_t)host_columns[index];
    }
    std::free(host_columns);

    status = map_cuda_status(cudaMalloc((void **)&columns16,
                                        matrix->nnz * sizeof(uint16_t)));
    if (status != HPRQP_STATUS_OK) {
        std::free(host_columns16);
        return status;
    }
    status = map_cuda_status(cudaMemcpy(columns16,
                                        host_columns16,
                                        matrix->nnz * sizeof(uint16_t),
                                        cudaMemcpyHostToDevice));
    std::free(host_columns16);
    if (status != HPRQP_STATUS_OK) {
        cudaFree(columns16);
        return status;
    }
    *device_columns = columns16;
    return HPRQP_STATUS_OK;
}

/*
 * Encode each compressible row as an exact uint16 offset from that row's
 * minimum column.  A negative base marks a fallback row, which continues to
 * read the original int32 column array.  Entry order is never changed.
 */
static HPRQPStatus gpu_parallel_convert_columns_to_row_delta_u16(
    const HPRQPGPUMatrixCSR *matrix,
    uint16_t **device_offsets,
    HPRQPGPUCsrIndex **device_bases,
    double *compressed_fraction)
{
    HPRQPGPUCsrIndex *host_row_ptr = NULL;
    HPRQPGPUCsrIndex *host_columns = NULL;
    HPRQPGPUCsrIndex *host_bases = NULL;
    uint16_t *host_offsets = NULL;
    HPRQPGPUCsrIndex *bases = NULL;
    uint16_t *offsets = NULL;
    size_t compressed_nnz = 0U;
    HPRQPStatus status = HPRQP_STATUS_OK;

    if (matrix == NULL || device_offsets == NULL ||
        device_bases == NULL || compressed_fraction == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *device_offsets = NULL;
    *device_bases = NULL;
    *compressed_fraction = 0.0;

    host_row_ptr = (HPRQPGPUCsrIndex *)std::malloc(
        (matrix->nrow + 1U) * sizeof(HPRQPGPUCsrIndex));
    host_bases = (HPRQPGPUCsrIndex *)std::malloc(
        matrix->nrow * sizeof(HPRQPGPUCsrIndex));
    if (matrix->nnz > 0U) {
        host_columns = (HPRQPGPUCsrIndex *)std::malloc(
            matrix->nnz * sizeof(HPRQPGPUCsrIndex));
        host_offsets = (uint16_t *)std::malloc(
            matrix->nnz * sizeof(uint16_t));
    }
    if (host_row_ptr == NULL ||
        (matrix->nrow > 0U && host_bases == NULL) ||
        (matrix->nnz > 0U &&
         (host_columns == NULL || host_offsets == NULL))) {
        status = HPRQP_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    status = map_cuda_status(cudaMemcpy(
        host_row_ptr,
        matrix->row_ptr,
        (matrix->nrow + 1U) * sizeof(HPRQPGPUCsrIndex),
        cudaMemcpyDeviceToHost));
    if (status != HPRQP_STATUS_OK) goto cleanup;
    if (matrix->nnz > 0U) {
        status = map_cuda_status(cudaMemcpy(
            host_columns,
            matrix->col_ind,
            matrix->nnz * sizeof(HPRQPGPUCsrIndex),
            cudaMemcpyDeviceToHost));
        if (status != HPRQP_STATUS_OK) goto cleanup;
    }

    for (size_t row = 0U; row < matrix->nrow; ++row) {
        const size_t begin = (size_t)host_row_ptr[row];
        const size_t end = (size_t)host_row_ptr[row + 1U];

        if (begin > end || end > matrix->nnz) {
            status = HPRQP_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        if (begin == end) {
            host_bases[row] = 0;
            continue;
        }
        HPRQPGPUCsrIndex minimum = host_columns[begin];
        HPRQPGPUCsrIndex maximum = host_columns[begin];
        for (size_t index = begin + 1U; index < end; ++index) {
            const HPRQPGPUCsrIndex column = host_columns[index];
            if (column < minimum) minimum = column;
            if (column > maximum) maximum = column;
        }
        if (minimum >= 0 && maximum >= minimum &&
            (uint32_t)(maximum - minimum) <= 65535U) {
            host_bases[row] = minimum;
            for (size_t index = begin; index < end; ++index) {
                host_offsets[index] =
                    (uint16_t)(host_columns[index] - minimum);
            }
            compressed_nnz += end - begin;
        } else {
            host_bases[row] = -1;
        }
    }

    if (matrix->nrow > 0U) {
        status = map_cuda_status(cudaMalloc(
            (void **)&bases,
            matrix->nrow * sizeof(HPRQPGPUCsrIndex)));
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = map_cuda_status(cudaMemcpy(
            bases,
            host_bases,
            matrix->nrow * sizeof(HPRQPGPUCsrIndex),
            cudaMemcpyHostToDevice));
        if (status != HPRQP_STATUS_OK) goto cleanup;
    }
    if (matrix->nnz > 0U) {
        status = map_cuda_status(cudaMalloc(
            (void **)&offsets,
            matrix->nnz * sizeof(uint16_t)));
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = map_cuda_status(cudaMemcpy(
            offsets,
            host_offsets,
            matrix->nnz * sizeof(uint16_t),
            cudaMemcpyHostToDevice));
        if (status != HPRQP_STATUS_OK) goto cleanup;
    }

    *device_offsets = offsets;
    *device_bases = bases;
    *compressed_fraction = matrix->nnz > 0U
                               ? (double)compressed_nnz /
                                     (double)matrix->nnz
                               : 1.0;
    offsets = NULL;
    bases = NULL;

cleanup:
    if (offsets != NULL) cudaFree(offsets);
    if (bases != NULL) cudaFree(bases);
    std::free(host_offsets);
    std::free(host_bases);
    std::free(host_columns);
    std::free(host_row_ptr);
    return status;
}

static HPRQPStatus gpu_parallel_prepare_col16(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem)
{
    double q_fraction = 0.0;
    double a_fraction = 0.0;
    double at_fraction = 0.0;
    HPRQPStatus status;

    if (workspace == NULL || problem == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    workspace->parallel_col16_available = 0;
    workspace->parallel_col16_delta_available = 0;
    workspace->parallel_persistent_use_col16 = 0;
    if (workspace->parallel_q_col16 != NULL) {
        cudaFree(workspace->parallel_q_col16);
        workspace->parallel_q_col16 = NULL;
    }
    if (workspace->parallel_a_col16 != NULL) {
        cudaFree(workspace->parallel_a_col16);
        workspace->parallel_a_col16 = NULL;
    }
    if (workspace->parallel_at_col16 != NULL) {
        cudaFree(workspace->parallel_at_col16);
        workspace->parallel_at_col16 = NULL;
    }
    if (workspace->parallel_q_col16_base != NULL) {
        cudaFree(workspace->parallel_q_col16_base);
        workspace->parallel_q_col16_base = NULL;
    }
    if (workspace->parallel_a_col16_base != NULL) {
        cudaFree(workspace->parallel_a_col16_base);
        workspace->parallel_a_col16_base = NULL;
    }
    if (workspace->parallel_at_col16_base != NULL) {
        cudaFree(workspace->parallel_at_col16_base);
        workspace->parallel_at_col16_base = NULL;
    }

    if (problem->Q.ncol <= 65536U && problem->A.ncol <= 65536U &&
        problem->AT.ncol <= 65536U) {
        status = gpu_parallel_convert_columns_to_u16(
            &problem->Q,
            &workspace->parallel_q_col16);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_parallel_convert_columns_to_u16(
            &problem->A,
            &workspace->parallel_a_col16);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_parallel_convert_columns_to_u16(
            &problem->AT,
            &workspace->parallel_at_col16);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        workspace->parallel_col16_available = 1;
        return HPRQP_STATUS_OK;
    }

    status = gpu_parallel_convert_columns_to_row_delta_u16(
        &problem->Q,
        &workspace->parallel_q_col16,
        &workspace->parallel_q_col16_base,
        &q_fraction);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_parallel_convert_columns_to_row_delta_u16(
        &problem->A,
        &workspace->parallel_a_col16,
        &workspace->parallel_a_col16_base,
        &a_fraction);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_parallel_convert_columns_to_row_delta_u16(
        &problem->AT,
        &workspace->parallel_at_col16,
        &workspace->parallel_at_col16_base,
        &at_fraction);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    {
        const double compressed_nnz =
            q_fraction * (double)problem->Q.nnz +
            a_fraction * (double)problem->A.nnz +
            at_fraction * (double)problem->AT.nnz;
        const double base_entries =
            (double)(problem->Q.nrow + problem->A.nrow +
                     problem->AT.nrow);

        /* Require a positive estimated index-byte saving before selection. */
        workspace->parallel_col16_delta_available =
            2.0 * compressed_nnz > 4.0 * base_entries;
    }
    return HPRQP_STATUS_OK;

cleanup:
    if (workspace->parallel_q_col16 != NULL) {
        cudaFree(workspace->parallel_q_col16);
        workspace->parallel_q_col16 = NULL;
    }
    if (workspace->parallel_a_col16 != NULL) {
        cudaFree(workspace->parallel_a_col16);
        workspace->parallel_a_col16 = NULL;
    }
    if (workspace->parallel_at_col16 != NULL) {
        cudaFree(workspace->parallel_at_col16);
        workspace->parallel_at_col16 = NULL;
    }
    if (workspace->parallel_q_col16_base != NULL) {
        cudaFree(workspace->parallel_q_col16_base);
        workspace->parallel_q_col16_base = NULL;
    }
    if (workspace->parallel_a_col16_base != NULL) {
        cudaFree(workspace->parallel_a_col16_base);
        workspace->parallel_a_col16_base = NULL;
    }
    if (workspace->parallel_at_col16_base != NULL) {
        cudaFree(workspace->parallel_at_col16_base);
        workspace->parallel_at_col16_base = NULL;
    }
    workspace->parallel_col16_available = 0;
    workspace->parallel_col16_delta_available = 0;
    return status;
}

static HPRQPStatus gpu_parallel_vector_all_positive_zero(
    const HPRQPGPUVector *vector,
    int *all_positive_zero)
{
    double *host_values = NULL;
    const double positive_zero = 0.0;
    HPRQPStatus status;

    if (vector == NULL || all_positive_zero == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *all_positive_zero = 1;
    if (vector->length == 0U) {
        return HPRQP_STATUS_OK;
    }
    host_values = (double *)std::malloc(vector->length * sizeof(double));
    if (host_values == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    status = map_cuda_status(cudaMemcpy(host_values,
                                        vector->values,
                                        vector->length * sizeof(double),
                                        cudaMemcpyDeviceToHost));
    if (status != HPRQP_STATUS_OK) {
        std::free(host_values);
        return status;
    }
    for (size_t index = 0U; index < vector->length; ++index) {
        if (std::memcmp(&host_values[index],
                        &positive_zero,
                        sizeof(double)) != 0) {
            *all_positive_zero = 0;
            break;
        }
    }
    std::free(host_values);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_parallel_csr_has_dominant_row_length(
    const HPRQPGPUMatrixCSR *matrix,
    HPRQPGPUCsrIndex target_length,
    int *is_dominant)
{
    HPRQPGPUCsrIndex *host_row_ptr = NULL;
    size_t matching_rows = 0U;
    HPRQPStatus status;

    if (matrix == NULL || is_dominant == NULL || target_length < 0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *is_dominant = 0;
    if (matrix->nrow == 0U) {
        return HPRQP_STATUS_OK;
    }
    host_row_ptr = (HPRQPGPUCsrIndex *)std::malloc(
        (matrix->nrow + 1U) * sizeof(HPRQPGPUCsrIndex));
    if (host_row_ptr == NULL) {
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }
    status = map_cuda_status(cudaMemcpy(
        host_row_ptr,
        matrix->row_ptr,
        (matrix->nrow + 1U) * sizeof(HPRQPGPUCsrIndex),
        cudaMemcpyDeviceToHost));
    if (status != HPRQP_STATUS_OK) {
        std::free(host_row_ptr);
        return status;
    }
    for (size_t row = 0U; row < matrix->nrow; ++row) {
        if (host_row_ptr[row + 1U] - host_row_ptr[row] ==
            target_length) {
            matching_rows += 1U;
        }
    }
    std::free(host_row_ptr);

    /* Require at least 90 percent so the fixed path remains predictable. */
    *is_dominant =
        matching_rows >= matrix->nrow - matrix->nrow / 10U;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_parallel_persistent_config_for_threads(
    unsigned int threads,
    unsigned int blocks_per_sm_limit,
    unsigned int *blocks,
    HPRQPGPUParallelPersistentKernel kernel_mode,
    size_t dynamic_shared_bytes,
    unsigned int fixed_row_length = 0U)
{
    cudaDeviceProp properties;
    const void *kernel;
    int device = 0;
    int active_blocks_per_sm = 0;
    cudaError_t cuda_status;

    if (blocks == NULL || threads == 0U || (threads & 31U) != 0U ||
        kernel_mode < HPRQP_GPU_PARALLEL_PERSISTENT_ROW ||
        kernel_mode > HPRQP_GPU_PARALLEL_PERSISTENT_GROUP) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    *blocks = 0U;
    cuda_status = cudaGetDevice(&device);
    if (cuda_status != cudaSuccess) {
        return map_cuda_status(cuda_status);
    }
    cuda_status = cudaGetDeviceProperties(&properties, device);
    if (cuda_status != cudaSuccess) {
        return map_cuda_status(cuda_status);
    }
    if (!properties.cooperativeLaunch) {
        return HPRQP_STATUS_OK;
    }
    if (dynamic_shared_bytes > properties.sharedMemPerBlock) {
        return HPRQP_STATUS_OK;
    }
    kernel = kernel_mode == HPRQP_GPU_PARALLEL_PERSISTENT_GROUP
                 ? (const void *)gpu_parallel_persistent_kernel
                 : kernel_mode ==
                           HPRQP_GPU_PARALLEL_PERSISTENT_ROW_PIPELINE
                       ? (const void *)gpu_parallel_persistent_row_pipeline_kernel
                       : fixed_row_length == 6U
                             ? (const void *)gpu_parallel_persistent_row_kernel<6U>
                             : fixed_row_length == 10U
                                   ? (const void *)gpu_parallel_persistent_row_kernel<10U>
                                   : (const void *)gpu_parallel_persistent_row_kernel<0U>;
    cuda_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        kernel,
        (int)threads,
        dynamic_shared_bytes);
    if (cuda_status != cudaSuccess) {
        return map_cuda_status(cuda_status);
    }
    if (blocks_per_sm_limit > 0U &&
        active_blocks_per_sm > (int)blocks_per_sm_limit) {
        active_blocks_per_sm = (int)blocks_per_sm_limit;
    }
    if (active_blocks_per_sm > 0) {
        *blocks = (unsigned int)active_blocks_per_sm *
                  (unsigned int)properties.multiProcessorCount;
    }
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_iteration_batch_parallel_config(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem,
    size_t inner_iter,
    size_t batch_count,
    unsigned int threads,
    unsigned int blocks,
    HPRQPGPUParallelPersistentKernel kernel_mode,
    int finish_check)
{
    HPRQPGPUParallelPersistentArgs args;
    double host_halpern[2U * HPRQP_GPU_PARALLEL_BATCH_MAX] = {0.0};
    size_t dynamic_shared_bytes = 0U;
    const int use_pipeline =
        kernel_mode == HPRQP_GPU_PARALLEL_PERSISTENT_ROW_PIPELINE;
    const int use_group =
        kernel_mode == HPRQP_GPU_PARALLEL_PERSISTENT_GROUP;
    void *kernel;
    void *kernel_args[2];
    HPRQPStatus status;
    int exact_single_diagonal_q;

    if (workspace == NULL || problem == NULL ||
        !workspace->allocated || !problem->uploaded ||
        batch_count == 0U ||
        batch_count > HPRQP_GPU_PARALLEL_BATCH_MAX ||
        threads == 0U || blocks == 0U ||
        kernel_mode < HPRQP_GPU_PARALLEL_PERSISTENT_ROW ||
        kernel_mode > HPRQP_GPU_PARALLEL_PERSISTENT_GROUP ||
        workspace->parallel_persistent_q_group_width < 32U ||
        workspace->parallel_persistent_q_group_width > threads ||
        (workspace->parallel_persistent_q_group_width & 31U) != 0U ||
        (use_pipeline &&
         workspace->parallel_persistent_q_group_width == threads) ||
        ((use_group ||
          kernel_mode == HPRQP_GPU_PARALLEL_PERSISTENT_ROW) &&
         threads % workspace->parallel_persistent_q_group_width != 0U)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (use_pipeline) {
        const size_t rows_per_block =
            workspace->n / (size_t)blocks +
            (workspace->n % (size_t)blocks != 0U ? 1U : 0U);

        dynamic_shared_bytes = rows_per_block * sizeof(double);
    }
    for (size_t i = 0; i < batch_count; ++i) {
        host_halpern[i] = 1.0 / ((double)(inner_iter + i) + 2.0);
        host_halpern[HPRQP_GPU_PARALLEL_BATCH_MAX + i] =
            1.0 - host_halpern[i];
    }
    status = map_cuda_status(cudaMemcpy(
        workspace->batch_halpern.values,
        host_halpern,
        (HPRQP_GPU_PARALLEL_BATCH_MAX + batch_count) *
            sizeof(double),
        cudaMemcpyHostToDevice));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    std::memset(&args, 0, sizeof(args));
    args.n = workspace->n;
    args.m = workspace->m;
    args.q_nonempty_count = workspace->parallel_q_nonempty_count;
    args.has_constraints = workspace->empty_A ? 0 : 1;
    args.q_nonempty_rows = workspace->parallel_q_nonempty_rows;
    args.q_row_ptr = problem->Q.row_ptr;
    args.q_col_ind = problem->Q.col_ind;
    args.q_col16 = workspace->parallel_q_col16;
    args.q_col16_base = workspace->parallel_q_col16_base;
    args.q_values = problem->Q.values;
    args.a_row_ptr = problem->A.row_ptr;
    args.a_col_ind = problem->A.col_ind;
    args.a_col16 = workspace->parallel_a_col16;
    args.a_col16_base = workspace->parallel_a_col16_base;
    args.a_values = problem->A.values;
    args.at_row_ptr = problem->AT.row_ptr;
    args.at_col_ind = problem->AT.col_ind;
    args.at_col16 = workspace->parallel_at_col16;
    args.at_col16_base = workspace->parallel_at_col16_base;
    args.at_values = problem->AT.values;
    args.c = problem->c.values;
    args.l = problem->l.values;
    args.u = problem->u.values;
    args.AL = problem->AL.values;
    args.AU = problem->AU.values;
    args.last_x = workspace->last_x.values;
    args.last_w = workspace->last_w.values;
    args.last_y = workspace->last_y.values;
    args.halpern_fact1 = workspace->batch_halpern.values;
    args.halpern_fact2 = workspace->batch_halpern_fact2.values;
    args.x = workspace->x.values;
    args.w = workspace->w.values;
    args.y = workspace->y.values;
    args.scratch_x = workspace->Qw_bar.values;
    args.scratch_w = workspace->ATy_bar.values;
    args.scratch_y = workspace->Rp.values;
    args.x_bar = workspace->x_bar.values;
    args.x_hat = workspace->x_hat.values;
    args.z_bar = workspace->z_bar.values;
    args.w_bar = workspace->w_bar.values;
    args.y_bar = workspace->y_bar.values;
    args.s = workspace->s.values;
    args.Qw = workspace->Qw.values;
    args.ATy = workspace->ATy.values;
    args.Ax = workspace->Ax.values;
    args.dx = workspace->dx.values;
    args.dw = workspace->dw.values;
    args.dy = workspace->dy.values;
    args.sigma = workspace->sigma;
    args.w_gamma = workspace->parallel_w_gamma;
    args.w_inv = workspace->parallel_w_inv;
    args.y_rho = workspace->parallel_y_rho;
    args.y_inv = workspace->parallel_y_inv;
    args.write_metric = finish_check;
    args.use_col16 = workspace->parallel_persistent_use_col16;
    args.x_lower_positive_zero =
        workspace->parallel_x_lower_positive_zero;
    args.constraints_positive_zero_equality =
        workspace->parallel_constraints_positive_zero_equality;
    args.q_group_width =
        workspace->parallel_persistent_q_group_width;

    kernel_args[0] = &args;
    kernel_args[1] = &batch_count;
    exact_single_diagonal_q =
        workspace->parallel_q_exact_single_diagonal &&
        (workspace->parallel_persistent_fixed_row_length != 0U ||
         workspace->parallel_q_nonempty_count >=
             (workspace->n + 1U) / 2U);
    if (use_group) {
        kernel = (void *)gpu_parallel_persistent_kernel;
    } else if (use_pipeline) {
        kernel = (void *)gpu_parallel_persistent_row_pipeline_kernel;
    } else if (workspace->parallel_persistent_fixed_row_length == 6U) {
        if (workspace->parallel_persistent_use_col16 == 2) {
            kernel = exact_single_diagonal_q
                         ? (void *)gpu_parallel_persistent_row_kernel<6U,
                                                                      2,
                                                                      true>
                         : (void *)gpu_parallel_persistent_row_kernel<6U, 2>;
        } else {
            kernel = exact_single_diagonal_q
                         ? (void *)gpu_parallel_persistent_row_kernel<6U,
                                                                      -1,
                                                                      true>
                         : (void *)gpu_parallel_persistent_row_kernel<6U>;
        }
    } else if (workspace->parallel_persistent_fixed_row_length == 10U) {
        kernel = exact_single_diagonal_q
                     ? (void *)gpu_parallel_persistent_row_kernel<10U,
                                                                  -1,
                                                                  true>
                     : (void *)gpu_parallel_persistent_row_kernel<10U>;
    } else {
        kernel = exact_single_diagonal_q
                     ? (void *)gpu_parallel_persistent_row_kernel<0U,
                                                                  -1,
                                                                  true>
                     : (void *)gpu_parallel_persistent_row_kernel<0U>;
    }
    status = map_cuda_status(cudaLaunchCooperativeKernel(
        kernel,
        dim3(blocks),
        dim3(threads),
        kernel_args,
        dynamic_shared_bytes,
        (cudaStream_t)0));
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (finish_check) {
        return gpu_parallel_finish_check(workspace, problem);
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_iteration_batch_parallel(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem,
    size_t inner_iter,
    size_t batch_count)
{
    if (workspace != NULL && problem != NULL && problem->p > 0) {
        HPRQPStatus status = HPRQP_STATUS_OK;
        const int finish_check = workspace->to_check;
        for (size_t offset = 0; offset < batch_count; ++offset) {
            workspace->to_check =
                finish_check && offset + 1U == batch_count;
            status = gpu_qcqp_iteration_step_parallel(
                workspace, problem, inner_iter + offset);
            if (status != HPRQP_STATUS_OK) {
                workspace->to_check = finish_check;
                return status;
            }
        }
        workspace->to_check = finish_check;
        return status;
    }
    if (workspace == NULL || !workspace->parallel_persistent_enabled) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    return gpu_iteration_batch_parallel_config(
        workspace,
        problem,
        inner_iter,
        batch_count,
        workspace->parallel_persistent_threads,
        workspace->parallel_persistent_blocks,
        workspace->parallel_persistent_kernel,
        workspace->to_check);
}

__global__ static void gpu_update_zxw1_full_kernel(double *dx,
                                                   double *z_bar,
                                                   double *w_bar,
                                                   double *x_bar,
                                                   double *x_hat,
                                                   double *x,
                                                   const double *last_x,
                                                   const double *w,
                                                   const double *Qw,
                                                   const double *ATy,
                                                   const double *c,
                                                   const double *l,
                                                   const double *u,
                                                   const double *fact1_vec,
                                                   const double *fact2_vec,
                                                   double sigma,
                                                   double fact1_scalar,
                                                   double fact2_scalar,
                                                   double halpern_fact1,
                                                   double halpern_fact2,
                                                   int q_is_diag,
                                                   size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double tmp = -Qw[i] + ATy[i] - c[i];
    double z_raw = x[i] + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x[i];
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w[i] + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w[i] + fact2_scalar * x_hat_i);

    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    dx[i] = x_bar_i - x[i];
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
}

__global__ static void gpu_update_zxw1_partial_kernel(double *w_bar,
                                                      double *x_hat,
                                                      double *x,
                                                      const double *last_x,
                                                      const double *w,
                                                      const double *Qw,
                                                      const double *ATy,
                                                      const double *c,
                                                      const double *l,
                                                      const double *u,
                                                      const double *fact1_vec,
                                                      const double *fact2_vec,
                                                      double sigma,
                                                      double fact1_scalar,
                                                      double fact2_scalar,
                                                      double halpern_fact1,
                                                      double halpern_fact2,
                                                      int q_is_diag,
                                                      size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double tmp = -Qw[i] + ATy[i] - c[i];
    double z_raw = x[i] + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x[i];
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w[i] + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w[i] + fact2_scalar * x_hat_i);

    x_hat[i] = x_hat_i;
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
}

__global__ static void gpu_update_zxw1_noC_full_kernel(double *dx,
                                                       double *dw,
                                                       double *z_bar,
                                                       double *w_bar,
                                                       double *x_bar,
                                                       double *x_hat,
                                                       double *x,
                                                       double *w,
                                                       const double *last_x,
                                                       const double *last_w,
                                                       const double *Qw,
                                                       const double *ATy,
                                                       const double *c,
                                                       const double *l,
                                                       const double *u,
                                                       const double *fact1_vec,
                                                       const double *fact2_vec,
                                                       double sigma,
                                                       double fact1_scalar,
                                                       double fact2_scalar,
                                                       double halpern_fact1,
                                                       double halpern_fact2,
                                                       int q_is_diag,
                                                       size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double x_i = x[i];
    double w_i = w[i];
    double tmp = -Qw[i] + ATy[i] - c[i];
    double z_raw = x_i + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x_i;
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w_i + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w_i + fact2_scalar * x_hat_i);
    double w_hat_i = 2.0 * w_bar_i - w_i;

    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    dx[i] = x_bar_i - x_i;
    dw[i] = w_bar_i - w_i;
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
    w[i] = halpern_fact2 * w_hat_i + halpern_fact1 * last_w[i];
}

__global__ static void gpu_update_zxw1_noC_partial_kernel(double *w_bar,
                                                          double *x_hat,
                                                          double *x,
                                                          double *w,
                                                          const double *last_x,
                                                          const double *last_w,
                                                          const double *Qw,
                                                          const double *ATy,
                                                          const double *c,
                                                          const double *l,
                                                          const double *u,
                                                          const double *fact1_vec,
                                                          const double *fact2_vec,
                                                          double sigma,
                                                          double fact1_scalar,
                                                          double fact2_scalar,
                                                          double halpern_fact1,
                                                          double halpern_fact2,
                                                          int q_is_diag,
                                                          size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double x_i = x[i];
    double w_i = w[i];
    double tmp = -Qw[i] + ATy[i] - c[i];
    double z_raw = x_i + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x_i;
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w_i + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w_i + fact2_scalar * x_hat_i);
    double w_hat_i = 2.0 * w_bar_i - w_i;

    x_hat[i] = x_hat_i;
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
    w[i] = halpern_fact2 * w_hat_i + halpern_fact1 * last_w[i];
}

__global__ static void gpu_update_zxw1_custom_full_kernel(double *dx,
                                                          double *z_bar,
                                                          double *w_bar,
                                                          double *x_bar,
                                                          double *x_hat,
                                                          double *x,
                                                          double *Qw,
                                                          const double *last_x,
                                                          const double *w,
                                                          const double *ATy,
                                                          const double *c,
                                                          const double *l,
                                                          const double *u,
                                                          const double *fact1_vec,
                                                          const double *fact2_vec,
                                                          const HPRQPGPUCsrIndex *row_ptr,
                                                          const HPRQPGPUCsrIndex *col_ind,
                                                          const double *values,
                                                          double sigma,
                                                          double fact1_scalar,
                                                          double fact2_scalar,
                                                          double halpern_fact1,
                                                          double halpern_fact2,
                                                          int q_is_diag,
                                                          size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_i = csr_row_dot_device(row_ptr, col_ind, values, w, i);
    double tmp = -qw_i + ATy[i] - c[i];
    double z_raw = x[i] + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x[i];
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w[i] + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w[i] + fact2_scalar * x_hat_i);

    Qw[i] = qw_i;
    x_bar[i] = x_bar_i;
    x_hat[i] = x_hat_i;
    z_bar[i] = (x_bar_i - z_raw) / sigma;
    dx[i] = x_bar_i - x[i];
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
}

__global__ static void gpu_update_zxw1_custom_partial_kernel(double *w_bar,
                                                             double *x_hat,
                                                             double *x,
                                                             double *Qw,
                                                             const double *last_x,
                                                             const double *w,
                                                             const double *ATy,
                                                             const double *c,
                                                             const double *l,
                                                             const double *u,
                                                             const double *fact1_vec,
                                                             const double *fact2_vec,
                                                             const HPRQPGPUCsrIndex *row_ptr,
                                                             const HPRQPGPUCsrIndex *col_ind,
                                                             const double *values,
                                                             double sigma,
                                                             double fact1_scalar,
                                                             double fact2_scalar,
                                                             double halpern_fact1,
                                                             double halpern_fact2,
                                                             int q_is_diag,
                                                             size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_i = csr_row_dot_device(row_ptr, col_ind, values, w, i);
    double tmp = -qw_i + ATy[i] - c[i];
    double z_raw = x[i] + sigma * tmp;
    double x_bar_i = interval_project_device(z_raw, l[i], u[i]);
    double x_hat_i = 2.0 * x_bar_i - x[i];
    double w_bar_i = q_is_diag ? (fact1_vec[i] * w[i] + fact2_vec[i] * x_hat_i)
                               : (fact1_scalar * w[i] + fact2_scalar * x_hat_i);

    Qw[i] = qw_i;
    x_hat[i] = x_hat_i;
    x[i] = halpern_fact2 * x_hat_i + halpern_fact1 * last_x[i];
    w_bar[i] = w_bar_i;
}

__global__ static void gpu_compute_tempv_kernel(double *tempv,
                                                const double *x_hat,
                                                const double *Qw,
                                                const double *Qw_bar,
                                                double sigma,
                                                size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    tempv[i] = x_hat[i] + sigma * (Qw[i] - Qw_bar[i]);
}

__global__ static void gpu_compute_tempv_custom_kernel(double *tempv,
                                                       const double *x_hat,
                                                       const double *w_bar,
                                                       const double *Qw,
                                                       const HPRQPGPUCsrIndex *row_ptr,
                                                       const HPRQPGPUCsrIndex *col_ind,
                                                       const double *values,
                                                       double sigma,
                                                       size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_bar_i = csr_row_dot_device(row_ptr, col_ind, values, w_bar, i);
    tempv[i] = x_hat[i] + sigma * (Qw[i] - qw_bar_i);
}

__global__ static void gpu_compute_tempv_custom_noc_kernel(double *tempv,
                                                           double *Qw_bar,
                                                           const double *x_hat,
                                                           const double *w_bar,
                                                           const double *Qw,
                                                           const HPRQPGPUCsrIndex *row_ptr,
                                                           const HPRQPGPUCsrIndex *col_ind,
                                                           const double *values,
                                                           double sigma,
                                                           size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_bar_i = csr_row_dot_device(row_ptr, col_ind, values, w_bar, i);
    Qw_bar[i] = qw_bar_i;
    tempv[i] = x_hat[i] + sigma * (Qw[i] - qw_bar_i);
}

__global__ static void gpu_finalize_qw_noC_kernel(double *tempv,
                                                  double *Qw,
                                                  const double *Qw_bar,
                                                  const double *last_Qw,
                                                  const double *x_hat,
                                                  double sigma,
                                                  double halpern_fact1,
                                                  double halpern_fact2,
                                                  size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double qw_i = Qw[i];
    double qw_bar_i = Qw_bar[i];
    double qw_hat_i = 2.0 * qw_bar_i - qw_i;

    Qw[i] = halpern_fact2 * qw_hat_i + halpern_fact1 * last_Qw[i];
    tempv[i] = x_hat[i] + sigma * (qw_i - qw_bar_i);
}

__global__ static void gpu_update_y_full_kernel(double *dy,
                                                double *y_bar,
                                                double *y,
                                                const double *last_y,
                                                double *s,
                                                const double *AL,
                                                const double *AU,
                                                const double *Ax,
                                                double fact1,
                                                double fact2,
                                                double halpern_fact1,
                                                double halpern_fact2,
                                                size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double s_raw = Ax[i] - fact1 * y[i];
    double s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    double y_bar_i = fact2 * (s_proj - s_raw);
    double y_hat_i = 2.0 * y_bar_i - y[i];

    y_bar[i] = y_bar_i;
    dy[i] = y_bar_i - y[i];
    s[i] = s_proj;
    y[i] = halpern_fact2 * y_hat_i + halpern_fact1 * last_y[i];
}

__global__ static void gpu_update_y_partial_kernel(double *y_bar,
                                                   double *y,
                                                   const double *last_y,
                                                   const double *AL,
                                                   const double *AU,
                                                   const double *Ax,
                                                   double fact1,
                                                   double fact2,
                                                   double halpern_fact1,
                                                   double halpern_fact2,
                                                   size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double s_raw = Ax[i] - fact1 * y[i];
    double s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    double y_bar_i = fact2 * (s_proj - s_raw);
    double y_hat_i = 2.0 * y_bar_i - y[i];

    y_bar[i] = y_bar_i;
    y[i] = halpern_fact2 * y_hat_i + halpern_fact1 * last_y[i];
}

__global__ static void gpu_update_y_custom_full_kernel(double *dy,
                                                       double *y_bar,
                                                       double *y,
                                                       double *s,
                                                       double *Ax,
                                                       const double *last_y,
                                                       const double *tempv,
                                                       const double *AL,
                                                       const double *AU,
                                                       const HPRQPGPUCsrIndex *row_ptr,
                                                       const HPRQPGPUCsrIndex *col_ind,
                                                       const double *values,
                                                       double fact1,
                                                       double fact2,
                                                       double halpern_fact1,
                                                       double halpern_fact2,
                                                       size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double ax_i = csr_row_dot_device(row_ptr, col_ind, values, tempv, i);
    double s_raw = ax_i - fact1 * y[i];
    double s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    double y_bar_i = fact2 * (s_proj - s_raw);
    double y_hat_i = 2.0 * y_bar_i - y[i];

    Ax[i] = ax_i;
    y_bar[i] = y_bar_i;
    dy[i] = y_bar_i - y[i];
    s[i] = s_proj;
    y[i] = halpern_fact2 * y_hat_i + halpern_fact1 * last_y[i];
}

__global__ static void gpu_update_y_custom_partial_kernel(double *y_bar,
                                                          double *y,
                                                          const double *last_y,
                                                          const double *tempv,
                                                          const double *AL,
                                                          const double *AU,
                                                          const HPRQPGPUCsrIndex *row_ptr,
                                                          const HPRQPGPUCsrIndex *col_ind,
                                                          const double *values,
                                                          double fact1,
                                                          double fact2,
                                                          double halpern_fact1,
                                                          double halpern_fact2,
                                                          size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double ax_i = csr_row_dot_device(row_ptr, col_ind, values, tempv, i);
    double s_raw = ax_i - fact1 * y[i];
    double s_proj = interval_project_device(s_raw, AL[i], AU[i]);
    double y_bar_i = fact2 * (s_proj - s_raw);
    double y_hat_i = 2.0 * y_bar_i - y[i];

    y_bar[i] = y_bar_i;
    y[i] = halpern_fact2 * y_hat_i + halpern_fact1 * last_y[i];
}

__global__ static void gpu_update_w2_full_kernel(double *dw,
                                                 double *ATdy,
                                                 double *w,
                                                 double *w_bar,
                                                 const double *last_w,
                                                 const double *ATy_bar,
                                                 const double *last_ATy,
                                                 double *ATy,
                                                 const double *fact_vec,
                                                 double fact_scalar,
                                                 double halpern_fact1,
                                                 double halpern_fact2,
                                                 int q_is_diag,
                                                 size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double fact;
    double w_bar_new;

    if (i >= n) {
        return;
    }

    fact = q_is_diag ? fact_vec[i] : fact_scalar;
    w_bar_new = w_bar[i] + fact * (ATy_bar[i] - ATy[i]);
    dw[i] = w_bar_new - w[i];
    ATdy[i] = ATy_bar[i] - ATy[i];
    w[i] = halpern_fact1 * last_w[i] + halpern_fact2 * (2.0 * w_bar_new - w[i]);
    ATy[i] = halpern_fact1 * last_ATy[i] + halpern_fact2 * (2.0 * ATy_bar[i] - ATy[i]);
    w_bar[i] = w_bar_new;
}

__global__ static void gpu_update_w2_partial_kernel(double *w,
                                                    const double *w_bar,
                                                    const double *last_w,
                                                    const double *ATy_bar,
                                                    const double *last_ATy,
                                                    double *ATy,
                                                    const double *fact_vec,
                                                    double fact_scalar,
                                                    double halpern_fact1,
                                                    double halpern_fact2,
                                                    int q_is_diag,
                                                    size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double fact;
    double w_bar_new;

    if (i >= n) {
        return;
    }

    fact = q_is_diag ? fact_vec[i] : fact_scalar;
    w_bar_new = w_bar[i] + fact * (ATy_bar[i] - ATy[i]);
    w[i] = halpern_fact1 * last_w[i] + halpern_fact2 * (2.0 * w_bar_new - w[i]);
    ATy[i] = halpern_fact1 * last_ATy[i] + halpern_fact2 * (2.0 * ATy_bar[i] - ATy[i]);
}

__global__ static void gpu_update_w2_custom_full_kernel(double *dw,
                                                        double *ATdy,
                                                        double *w,
                                                        double *w_bar,
                                                        double *ATy,
                                                        double *ATy_bar,
                                                        const double *last_w,
                                                        const double *last_ATy,
                                                        const double *y_bar,
                                                        const double *fact_vec,
                                                        const HPRQPGPUCsrIndex *row_ptr,
                                                        const HPRQPGPUCsrIndex *col_ind,
                                                        const double *values,
                                                        double fact_scalar,
                                                        double halpern_fact1,
                                                        double halpern_fact2,
                                                        int q_is_diag,
                                                        size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double aty_bar_i = csr_row_dot_device(row_ptr, col_ind, values, y_bar, i);
    double fact = q_is_diag ? fact_vec[i] : fact_scalar;
    double w_bar_new = w_bar[i] + fact * (aty_bar_i - ATy[i]);

    ATy_bar[i] = aty_bar_i;
    dw[i] = w_bar_new - w[i];
    ATdy[i] = aty_bar_i - ATy[i];
    w[i] = halpern_fact1 * last_w[i] + halpern_fact2 * (2.0 * w_bar_new - w[i]);
    ATy[i] = halpern_fact1 * last_ATy[i] + halpern_fact2 * (2.0 * aty_bar_i - ATy[i]);
    w_bar[i] = w_bar_new;
}

__global__ static void gpu_update_w2_custom_partial_kernel(double *w,
                                                           double *ATy,
                                                           const double *w_bar,
                                                           const double *last_w,
                                                           const double *last_ATy,
                                                           const double *y_bar,
                                                           const double *fact_vec,
                                                           const HPRQPGPUCsrIndex *row_ptr,
                                                           const HPRQPGPUCsrIndex *col_ind,
                                                           const double *values,
                                                           double fact_scalar,
                                                           double halpern_fact1,
                                                           double halpern_fact2,
                                                           int q_is_diag,
                                                           size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double aty_bar_i = csr_row_dot_device(row_ptr, col_ind, values, y_bar, i);
    double fact = q_is_diag ? fact_vec[i] : fact_scalar;
    double w_bar_new = w_bar[i] + fact * (aty_bar_i - ATy[i]);

    w[i] = halpern_fact1 * last_w[i] + halpern_fact2 * (2.0 * w_bar_new - w[i]);
    ATy[i] = halpern_fact1 * last_ATy[i] + halpern_fact2 * (2.0 * aty_bar_i - ATy[i]);
}

static HPRQPStatus gpu_update_zxw1_step(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPGPUProblem *gpu_problem,
                                        double halpern_fact1,
                                        double halpern_fact2,
                                        unsigned int threads_n,
                                        unsigned int blocks_n)
{
    double fact2_scalar = 1.0 / (1.0 + gpu_workspace->sigma * gpu_workspace->lambda_max_Q);
    double fact1_scalar = 1.0 - fact2_scalar;

    if (gpu_workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        if (gpu_workspace->to_check) {
            gpu_update_zxw1_custom_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dx.values,
                                                                        gpu_workspace->z_bar.values,
                                                                        gpu_workspace->w_bar.values,
                                                                        gpu_workspace->x_bar.values,
                                                                        gpu_workspace->x_hat.values,
                                                                        gpu_workspace->x.values,
                                                                        gpu_workspace->Qw.values,
                                                                        gpu_workspace->last_x.values,
                                                                        gpu_workspace->w.values,
                                                                        gpu_workspace->ATy.values,
                                                                        gpu_problem->c.values,
                                                                        gpu_problem->l.values,
                                                                        gpu_problem->u.values,
                                                                        gpu_workspace->fact1.values,
                                                                        gpu_workspace->fact2.values,
                                                                        gpu_problem->Q.row_ptr,
                                                                        gpu_problem->Q.col_ind,
                                                                        gpu_problem->Q.values,
                                                                        gpu_workspace->sigma,
                                                                        fact1_scalar,
                                                                        fact2_scalar,
                                                                        halpern_fact1,
                                                                        halpern_fact2,
                                                                        gpu_workspace->Q_is_diag,
                                                                        gpu_workspace->n);
        } else {
            gpu_update_zxw1_custom_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w_bar.values,
                                                                           gpu_workspace->x_hat.values,
                                                                           gpu_workspace->x.values,
                                                                           gpu_workspace->Qw.values,
                                                                           gpu_workspace->last_x.values,
                                                                           gpu_workspace->w.values,
                                                                           gpu_workspace->ATy.values,
                                                                           gpu_problem->c.values,
                                                                           gpu_problem->l.values,
                                                                           gpu_problem->u.values,
                                                                           gpu_workspace->fact1.values,
                                                                           gpu_workspace->fact2.values,
                                                                           gpu_problem->Q.row_ptr,
                                                                           gpu_problem->Q.col_ind,
                                                                           gpu_problem->Q.values,
                                                                           gpu_workspace->sigma,
                                                                           fact1_scalar,
                                                                           fact2_scalar,
                                                                           halpern_fact1,
                                                                           halpern_fact2,
                                                                           gpu_workspace->Q_is_diag,
                                                                           gpu_workspace->n);
        }
    } else {
        (void)gpu_spmv_apply_q_w(gpu_workspace, gpu_problem);
        if (gpu_workspace->to_check) {
            gpu_update_zxw1_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dx.values,
                                                                 gpu_workspace->z_bar.values,
                                                                 gpu_workspace->w_bar.values,
                                                                 gpu_workspace->x_bar.values,
                                                                 gpu_workspace->x_hat.values,
                                                                 gpu_workspace->x.values,
                                                                 gpu_workspace->last_x.values,
                                                                 gpu_workspace->w.values,
                                                                 gpu_workspace->Qw.values,
                                                                 gpu_workspace->ATy.values,
                                                                 gpu_problem->c.values,
                                                                 gpu_problem->l.values,
                                                                 gpu_problem->u.values,
                                                                 gpu_workspace->fact1.values,
                                                                 gpu_workspace->fact2.values,
                                                                 gpu_workspace->sigma,
                                                                 fact1_scalar,
                                                                 fact2_scalar,
                                                                 halpern_fact1,
                                                                 halpern_fact2,
                                                                 gpu_workspace->Q_is_diag,
                                                                 gpu_workspace->n);
        } else {
            gpu_update_zxw1_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w_bar.values,
                                                                    gpu_workspace->x_hat.values,
                                                                    gpu_workspace->x.values,
                                                                    gpu_workspace->last_x.values,
                                                                    gpu_workspace->w.values,
                                                                    gpu_workspace->Qw.values,
                                                                    gpu_workspace->ATy.values,
                                                                    gpu_problem->c.values,
                                                                    gpu_problem->l.values,
                                                                    gpu_problem->u.values,
                                                                    gpu_workspace->fact1.values,
                                                                    gpu_workspace->fact2.values,
                                                                    gpu_workspace->sigma,
                                                                    fact1_scalar,
                                                                    fact2_scalar,
                                                                    halpern_fact1,
                                                                    halpern_fact2,
                                                                    gpu_workspace->Q_is_diag,
                                                                    gpu_workspace->n);
        }
    }

    if (gpu_workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        gpu_compute_tempv_custom_kernel<<<blocks_n, threads_n>>>(gpu_workspace->tempv.values,
                                                                 gpu_workspace->x_hat.values,
                                                                 gpu_workspace->w_bar.values,
                                                                 gpu_workspace->Qw.values,
                                                                 gpu_problem->Q.row_ptr,
                                                                 gpu_problem->Q.col_ind,
                                                                 gpu_problem->Q.values,
                                                                 gpu_workspace->sigma,
                                                                 gpu_workspace->n);
    } else {
        (void)gpu_spmv_apply_q_w_bar(gpu_workspace, gpu_problem);
        gpu_compute_tempv_kernel<<<blocks_n, threads_n>>>(gpu_workspace->tempv.values,
                                                          gpu_workspace->x_hat.values,
                                                          gpu_workspace->Qw.values,
                                                          gpu_workspace->Qw_bar.values,
                                                          gpu_workspace->sigma,
                                                          gpu_workspace->n);
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_zxw1_noC_step(HPRQPGPUWorkspace *gpu_workspace,
                                            const HPRQPGPUProblem *gpu_problem,
                                            double halpern_fact1,
                                            double halpern_fact2,
                                            unsigned int threads_n,
                                            unsigned int blocks_n)
{
    double fact2_scalar = 1.0 / (1.0 + gpu_workspace->sigma * gpu_workspace->lambda_max_Q);
    double fact1_scalar = 1.0 - fact2_scalar;

    if (!gpu_workspace->empty_A) {
        (void)gpu_spmv_apply_at_y(gpu_workspace, gpu_problem);
    } else if (gpu_workspace->ATy.length > 0) {
        cudaMemset(gpu_workspace->ATy.values, 0, gpu_workspace->ATy.length * sizeof(double));
    }

    if (gpu_workspace->to_check) {
        gpu_update_zxw1_noC_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dx.values,
                                                                 gpu_workspace->dw.values,
                                                                 gpu_workspace->z_bar.values,
                                                                 gpu_workspace->w_bar.values,
                                                                 gpu_workspace->x_bar.values,
                                                                 gpu_workspace->x_hat.values,
                                                                 gpu_workspace->x.values,
                                                                 gpu_workspace->w.values,
                                                                 gpu_workspace->last_x.values,
                                                                 gpu_workspace->last_w.values,
                                                                 gpu_workspace->Qw.values,
                                                                 gpu_workspace->ATy.values,
                                                                 gpu_problem->c.values,
                                                                 gpu_problem->l.values,
                                                                 gpu_problem->u.values,
                                                                 gpu_workspace->fact1.values,
                                                                 gpu_workspace->fact2.values,
                                                                 gpu_workspace->sigma,
                                                                 fact1_scalar,
                                                                 fact2_scalar,
                                                                 halpern_fact1,
                                                                 halpern_fact2,
                                                                 gpu_workspace->Q_is_diag,
                                                                 gpu_workspace->n);
    } else {
        gpu_update_zxw1_noC_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w_bar.values,
                                                                    gpu_workspace->x_hat.values,
                                                                    gpu_workspace->x.values,
                                                                    gpu_workspace->w.values,
                                                                    gpu_workspace->last_x.values,
                                                                    gpu_workspace->last_w.values,
                                                                    gpu_workspace->Qw.values,
                                                                    gpu_workspace->ATy.values,
                                                                    gpu_problem->c.values,
                                                                    gpu_problem->l.values,
                                                                    gpu_problem->u.values,
                                                                    gpu_workspace->fact1.values,
                                                                    gpu_workspace->fact2.values,
                                                                    gpu_workspace->sigma,
                                                                    fact1_scalar,
                                                                    fact2_scalar,
                                                                    halpern_fact1,
                                                                    halpern_fact2,
                                                                    gpu_workspace->Q_is_diag,
                                                                    gpu_workspace->n);
    }

    if (gpu_workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED) {
        gpu_compute_tempv_custom_noc_kernel<<<blocks_n, threads_n>>>(gpu_workspace->tempv.values,
                                                                     gpu_workspace->Qw_bar.values,
                                                                     gpu_workspace->x_hat.values,
                                                                     gpu_workspace->w_bar.values,
                                                                     gpu_workspace->Qw.values,
                                                                     gpu_problem->Q.row_ptr,
                                                                     gpu_problem->Q.col_ind,
                                                                     gpu_problem->Q.values,
                                                                     gpu_workspace->sigma,
                                                                     gpu_workspace->n);
    } else {
        (void)gpu_spmv_apply_q_w_bar(gpu_workspace, gpu_problem);
    }

    gpu_finalize_qw_noC_kernel<<<blocks_n, threads_n>>>(gpu_workspace->tempv.values,
                                                        gpu_workspace->Qw.values,
                                                        gpu_workspace->Qw_bar.values,
                                                        gpu_workspace->last_Qw.values,
                                                        gpu_workspace->x_hat.values,
                                                        gpu_workspace->sigma,
                                                        halpern_fact1,
                                                        halpern_fact2,
                                                        gpu_workspace->n);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_y_step(HPRQPGPUWorkspace *gpu_workspace,
                                     const HPRQPGPUProblem *gpu_problem,
                                     double halpern_fact1,
                                     double halpern_fact2,
                                     unsigned int threads_m,
                                     unsigned int blocks_m)
{
    double fact1;
    double fact2;

    if (gpu_workspace->empty_A || threads_m == 0U) {
        return HPRQP_STATUS_OK;
    }

    fact1 = gpu_workspace->lambda_max_A * gpu_workspace->sigma;
    fact2 = 1.0 / fact1;

    if (gpu_workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        if (gpu_workspace->to_check) {
            gpu_update_y_custom_full_kernel<<<blocks_m, threads_m>>>(gpu_workspace->dy.values,
                                                                     gpu_workspace->y_bar.values,
                                                                     gpu_workspace->y.values,
                                                                     gpu_workspace->s.values,
                                                                     gpu_workspace->Ax.values,
                                                                     gpu_workspace->last_y.values,
                                                                     gpu_workspace->tempv.values,
                                                                     gpu_problem->AL.values,
                                                                     gpu_problem->AU.values,
                                                                     gpu_problem->A.row_ptr,
                                                                     gpu_problem->A.col_ind,
                                                                     gpu_problem->A.values,
                                                                     fact1,
                                                                     fact2,
                                                                     halpern_fact1,
                                                                     halpern_fact2,
                                                                     gpu_workspace->m);
        } else {
            gpu_update_y_custom_partial_kernel<<<blocks_m, threads_m>>>(gpu_workspace->y_bar.values,
                                                                        gpu_workspace->y.values,
                                                                        gpu_workspace->last_y.values,
                                                                        gpu_workspace->tempv.values,
                                                                        gpu_problem->AL.values,
                                                                        gpu_problem->AU.values,
                                                                        gpu_problem->A.row_ptr,
                                                                        gpu_problem->A.col_ind,
                                                                        gpu_problem->A.values,
                                                                        fact1,
                                                                        fact2,
                                                                        halpern_fact1,
                                                                        halpern_fact2,
                                                                        gpu_workspace->m);
        }
    } else {
        (void)gpu_spmv_apply_a_tempv(gpu_workspace, gpu_problem);
        if (gpu_workspace->to_check) {
            gpu_update_y_full_kernel<<<blocks_m, threads_m>>>(gpu_workspace->dy.values,
                                                              gpu_workspace->y_bar.values,
                                                              gpu_workspace->y.values,
                                                              gpu_workspace->last_y.values,
                                                              gpu_workspace->s.values,
                                                              gpu_problem->AL.values,
                                                              gpu_problem->AU.values,
                                                              gpu_workspace->Ax.values,
                                                              fact1,
                                                              fact2,
                                                              halpern_fact1,
                                                              halpern_fact2,
                                                              gpu_workspace->m);
        } else {
            gpu_update_y_partial_kernel<<<blocks_m, threads_m>>>(gpu_workspace->y_bar.values,
                                                                 gpu_workspace->y.values,
                                                                 gpu_workspace->last_y.values,
                                                                 gpu_problem->AL.values,
                                                                 gpu_problem->AU.values,
                                                                 gpu_workspace->Ax.values,
                                                                 fact1,
                                                                 fact2,
                                                                 halpern_fact1,
                                                                 halpern_fact2,
                                                                 gpu_workspace->m);
        }
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_update_w2_step(HPRQPGPUWorkspace *gpu_workspace,
                                      const HPRQPGPUProblem *gpu_problem,
                                      double halpern_fact1,
                                      double halpern_fact2,
                                      unsigned int threads_n,
                                      unsigned int blocks_n)
{
    double fact_scalar;

    fact_scalar = gpu_workspace->sigma / (1.0 + gpu_workspace->sigma * gpu_workspace->lambda_max_Q);

    if (gpu_workspace->empty_A) {
        if (gpu_workspace->to_check) {
            gpu_update_w2_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dw.values,
                                                               gpu_workspace->ATdy.values,
                                                               gpu_workspace->w.values,
                                                               gpu_workspace->w_bar.values,
                                                               gpu_workspace->last_w.values,
                                                               gpu_workspace->ATy_bar.values,
                                                               gpu_workspace->last_ATy.values,
                                                               gpu_workspace->ATy.values,
                                                               gpu_workspace->fact.values,
                                                               fact_scalar,
                                                               halpern_fact1,
                                                               halpern_fact2,
                                                               gpu_workspace->Q_is_diag,
                                                               gpu_workspace->n);
        } else {
            gpu_update_w2_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w.values,
                                                                  gpu_workspace->w_bar.values,
                                                                  gpu_workspace->last_w.values,
                                                                  gpu_workspace->ATy_bar.values,
                                                                  gpu_workspace->last_ATy.values,
                                                                  gpu_workspace->ATy.values,
                                                                  gpu_workspace->fact.values,
                                                                  fact_scalar,
                                                                  halpern_fact1,
                                                                  halpern_fact2,
                                                                  gpu_workspace->Q_is_diag,
                                                                  gpu_workspace->n);
        }
        return HPRQP_STATUS_OK;
    }

    if (gpu_workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED) {
        if (gpu_workspace->to_check) {
            gpu_update_w2_custom_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dw.values,
                                                                      gpu_workspace->ATdy.values,
                                                                      gpu_workspace->w.values,
                                                                      gpu_workspace->w_bar.values,
                                                                      gpu_workspace->ATy.values,
                                                                      gpu_workspace->ATy_bar.values,
                                                                      gpu_workspace->last_w.values,
                                                                      gpu_workspace->last_ATy.values,
                                                                      gpu_workspace->y_bar.values,
                                                                      gpu_workspace->fact.values,
                                                                      gpu_problem->AT.row_ptr,
                                                                      gpu_problem->AT.col_ind,
                                                                      gpu_problem->AT.values,
                                                                      fact_scalar,
                                                                      halpern_fact1,
                                                                      halpern_fact2,
                                                                      gpu_workspace->Q_is_diag,
                                                                      gpu_workspace->n);
        } else {
            gpu_update_w2_custom_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w.values,
                                                                         gpu_workspace->ATy.values,
                                                                         gpu_workspace->w_bar.values,
                                                                         gpu_workspace->last_w.values,
                                                                         gpu_workspace->last_ATy.values,
                                                                         gpu_workspace->y_bar.values,
                                                                         gpu_workspace->fact.values,
                                                                         gpu_problem->AT.row_ptr,
                                                                         gpu_problem->AT.col_ind,
                                                                         gpu_problem->AT.values,
                                                                         fact_scalar,
                                                                         halpern_fact1,
                                                                         halpern_fact2,
                                                                         gpu_workspace->Q_is_diag,
                                                                         gpu_workspace->n);
        }
    } else {
        (void)gpu_spmv_apply_at_y_bar(gpu_workspace, gpu_problem);
        if (gpu_workspace->to_check) {
            gpu_update_w2_full_kernel<<<blocks_n, threads_n>>>(gpu_workspace->dw.values,
                                                               gpu_workspace->ATdy.values,
                                                               gpu_workspace->w.values,
                                                               gpu_workspace->w_bar.values,
                                                               gpu_workspace->last_w.values,
                                                               gpu_workspace->ATy_bar.values,
                                                               gpu_workspace->last_ATy.values,
                                                               gpu_workspace->ATy.values,
                                                               gpu_workspace->fact.values,
                                                               fact_scalar,
                                                               halpern_fact1,
                                                               halpern_fact2,
                                                               gpu_workspace->Q_is_diag,
                                                               gpu_workspace->n);
        } else {
            gpu_update_w2_partial_kernel<<<blocks_n, threads_n>>>(gpu_workspace->w.values,
                                                                  gpu_workspace->w_bar.values,
                                                                  gpu_workspace->last_w.values,
                                                                  gpu_workspace->ATy_bar.values,
                                                                  gpu_workspace->last_ATy.values,
                                                                  gpu_workspace->ATy.values,
                                                                  gpu_workspace->fact.values,
                                                                  fact_scalar,
                                                                  halpern_fact1,
                                                                  halpern_fact2,
                                                                  gpu_workspace->Q_is_diag,
                                                                  gpu_workspace->n);
        }
    }

    return HPRQP_STATUS_OK;
}

static HPRQPStatus benchmark_q_mode(double *elapsed_ms,
                                    HPRQPGPUWorkspace *gpu_workspace,
                                    const HPRQPGPUProblem *gpu_problem,
                                    HPRQPGPUWorkspace *benchmark_workspace,
                                    HPRQPSpmvMode mode)
{
    cudaEvent_t start_event = NULL;
    cudaEvent_t stop_event = NULL;
    HPRQPSpmvMode saved_mode = gpu_workspace->spmv_mode_q;
    HPRQPStatus status;
    float elapsed_ms_float;
    size_t rep;
    unsigned int threads_n;
    unsigned int blocks_n;

    if (elapsed_ms == NULL || gpu_workspace == NULL || gpu_problem == NULL || benchmark_workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_workspace_copy_q_benchmark_state(benchmark_workspace, gpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    gpu_workspace->spmv_mode_q = mode;
    threads_n = gpu_workspace->n == 0 ? 0U : 256U;
    blocks_n = gpu_workspace->n == 0 ? 0U : (unsigned int)((gpu_workspace->n + threads_n - 1) / threads_n);

    if (cudaEventCreate(&start_event) != cudaSuccess || cudaEventCreate(&stop_event) != cudaSuccess) {
        if (start_event != NULL) {
            cudaEventDestroy(start_event);
        }
        if (stop_event != NULL) {
            cudaEventDestroy(stop_event);
        }
        (void)gpu_workspace_copy_q_benchmark_state(gpu_workspace, benchmark_workspace);
        gpu_workspace->spmv_mode_q = saved_mode;
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    cudaEventRecord(start_event);
    for (rep = 0; rep < kSpmvBenchmarkRepetitions; ++rep) {
        status = gpu_update_zxw1_step(gpu_workspace, gpu_problem, 0.5, 0.5, threads_n, blocks_n);
        if (status != HPRQP_STATUS_OK) {
            cudaEventDestroy(stop_event);
            cudaEventDestroy(start_event);
            (void)gpu_workspace_copy_q_benchmark_state(gpu_workspace, benchmark_workspace);
            gpu_workspace->spmv_mode_q = saved_mode;
            return status;
        }
    }
    cudaEventRecord(stop_event);
    if (cudaEventSynchronize(stop_event) != cudaSuccess || cudaEventElapsedTime(&elapsed_ms_float, start_event, stop_event) != cudaSuccess) {
        cudaEventDestroy(stop_event);
        cudaEventDestroy(start_event);
        (void)gpu_workspace_copy_q_benchmark_state(gpu_workspace, benchmark_workspace);
        gpu_workspace->spmv_mode_q = saved_mode;
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_event);
    status = gpu_workspace_copy_q_benchmark_state(gpu_workspace, benchmark_workspace);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace->spmv_mode_q = saved_mode;
        return status;
    }
    gpu_workspace->spmv_mode_q = saved_mode;
    *elapsed_ms = (double)elapsed_ms_float;
    return HPRQP_STATUS_OK;
}

static HPRQPStatus benchmark_a_mode(double *elapsed_ms,
                                    HPRQPGPUWorkspace *gpu_workspace,
                                    const HPRQPGPUProblem *gpu_problem,
                                    HPRQPGPUWorkspace *benchmark_workspace,
                                    HPRQPSpmvMode mode)
{
    cudaEvent_t start_event = NULL;
    cudaEvent_t stop_event = NULL;
    HPRQPSpmvMode saved_mode = gpu_workspace->spmv_mode_a;
    HPRQPStatus status;
    float elapsed_ms_float;
    size_t rep;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;

    if (elapsed_ms == NULL || gpu_workspace == NULL || gpu_problem == NULL || benchmark_workspace == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (gpu_workspace->empty_A) {
        *elapsed_ms = 0.0;
        return HPRQP_STATUS_OK;
    }

    status = gpu_workspace_copy_a_benchmark_state(benchmark_workspace, gpu_workspace);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    gpu_workspace->spmv_mode_a = mode;
    threads_n = gpu_workspace->n == 0 ? 0U : 256U;
    blocks_n = gpu_workspace->n == 0 ? 0U : (unsigned int)((gpu_workspace->n + threads_n - 1) / threads_n);
    threads_m = gpu_workspace->threads_m;
    blocks_m = gpu_workspace->blocks_m;

    if (cudaEventCreate(&start_event) != cudaSuccess || cudaEventCreate(&stop_event) != cudaSuccess) {
        if (start_event != NULL) {
            cudaEventDestroy(start_event);
        }
        if (stop_event != NULL) {
            cudaEventDestroy(stop_event);
        }
        (void)gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
        gpu_workspace->spmv_mode_a = saved_mode;
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    cudaEventRecord(start_event);
    for (rep = 0; rep < kSpmvBenchmarkRepetitions; ++rep) {
        status = gpu_update_y_step(gpu_workspace, gpu_problem, 0.5, 0.5, threads_m, blocks_m);
        if (status != HPRQP_STATUS_OK) {
            cudaEventDestroy(stop_event);
            cudaEventDestroy(start_event);
            (void)gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
            gpu_workspace->spmv_mode_a = saved_mode;
            return status;
        }
        status = gpu_update_w2_step(gpu_workspace, gpu_problem, 0.5, 0.5, threads_n, blocks_n);
        if (status != HPRQP_STATUS_OK) {
            cudaEventDestroy(stop_event);
            cudaEventDestroy(start_event);
            (void)gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
            gpu_workspace->spmv_mode_a = saved_mode;
            return status;
        }
    }
    cudaEventRecord(stop_event);
    if (cudaEventSynchronize(stop_event) != cudaSuccess || cudaEventElapsedTime(&elapsed_ms_float, start_event, stop_event) != cudaSuccess) {
        cudaEventDestroy(stop_event);
        cudaEventDestroy(start_event);
        (void)gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
        gpu_workspace->spmv_mode_a = saved_mode;
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    cudaEventDestroy(stop_event);
    cudaEventDestroy(start_event);
    status = gpu_workspace_copy_a_benchmark_state(gpu_workspace, benchmark_workspace);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace->spmv_mode_a = saved_mode;
        return status;
    }
    gpu_workspace->spmv_mode_a = saved_mode;
    *elapsed_ms = (double)elapsed_ms_float;
    return HPRQP_STATUS_OK;
}

typedef struct HPRQPParallelQKernelCandidate {
    HPRQPSpmvMode mode;
    HPRQPGPUCustomSpmvKernel custom_kernel;
    int fuse_q;
} HPRQPParallelQKernelCandidate;

typedef struct HPRQPParallelAKernelCandidate {
    HPRQPSpmvMode mode;
    HPRQPGPUCustomSpmvKernel custom_kernel;
    int fuse_at;
    int fuse_a;
} HPRQPParallelAKernelCandidate;

static HPRQPStatus gpu_workspace_copy_parallel_benchmark_state(
    HPRQPGPUWorkspace *dst,
    const HPRQPGPUWorkspace *src)
{
    HPRQPStatus status;

    status = gpu_workspace_copy_q_benchmark_state(dst, src);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    return gpu_workspace_copy_a_benchmark_state(dst, src);
}

static HPRQPStatus benchmark_parallel_kernel_candidate(
    double *elapsed_ms,
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *model,
    const HPRQPGPUWorkspace *saved_state,
    const HPRQPParallelQKernelCandidate *q_candidate,
    const HPRQPParallelAKernelCandidate *a_candidate)
{
    static const size_t warmup_repetitions = 8U;
    static const size_t timing_repetitions = 100U;
    HPRQPSpmvMode saved_mode_q;
    HPRQPSpmvMode saved_mode_a;
    HPRQPGPUCustomSpmvKernel saved_custom_q;
    HPRQPGPUCustomSpmvKernel saved_custom_a;
    int saved_fuse_q;
    int saved_fuse_at;
    int saved_fuse_a;
    int saved_to_check;
    HPRQPStatus status;
    size_t rep;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point stop;

    if (elapsed_ms == NULL || workspace == NULL || model == NULL ||
        saved_state == NULL || q_candidate == NULL ||
        a_candidate == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    saved_mode_q = workspace->spmv_mode_q;
    saved_mode_a = workspace->spmv_mode_a;
    saved_custom_q = workspace->custom_spmv_q_kernel;
    saved_custom_a = workspace->custom_spmv_a_kernel;
    saved_fuse_q = workspace->parallel_fuse_q;
    saved_fuse_at = workspace->parallel_fuse_at;
    saved_fuse_a = workspace->parallel_fuse_a;
    saved_to_check = workspace->to_check;

    status = gpu_workspace_copy_parallel_benchmark_state(workspace,
                                                         saved_state);
    if (status != HPRQP_STATUS_OK) goto restore_configuration;
    workspace->spmv_mode_q = q_candidate->mode;
    workspace->spmv_mode_a = a_candidate->mode;
    workspace->custom_spmv_q_kernel = q_candidate->custom_kernel;
    workspace->custom_spmv_a_kernel = a_candidate->custom_kernel;
    workspace->parallel_fuse_q = q_candidate->fuse_q;
    workspace->parallel_fuse_at = a_candidate->fuse_at;
    workspace->parallel_fuse_a = a_candidate->fuse_a;
    workspace->to_check = 0;
    for (rep = 0; rep < warmup_repetitions; ++rep) {
        status = gpu_iteration_step_parallel(workspace, model, rep);
        if (status != HPRQP_STATUS_OK) goto restore_state;
    }
    status = map_cuda_status(cudaDeviceSynchronize());
    if (status != HPRQP_STATUS_OK) goto restore_state;

    status = gpu_workspace_copy_parallel_benchmark_state(workspace,
                                                         saved_state);
    if (status != HPRQP_STATUS_OK) goto restore_configuration;
    workspace->spmv_mode_q = q_candidate->mode;
    workspace->spmv_mode_a = a_candidate->mode;
    workspace->custom_spmv_q_kernel = q_candidate->custom_kernel;
    workspace->custom_spmv_a_kernel = a_candidate->custom_kernel;
    workspace->parallel_fuse_q = q_candidate->fuse_q;
    workspace->parallel_fuse_at = a_candidate->fuse_at;
    workspace->parallel_fuse_a = a_candidate->fuse_a;
    workspace->to_check = 0;
    status = map_cuda_status(cudaDeviceSynchronize());
    if (status != HPRQP_STATUS_OK) goto restore_state;
    start = std::chrono::steady_clock::now();
    for (rep = 0; rep < timing_repetitions; ++rep) {
        status = gpu_iteration_step_parallel(workspace, model, rep);
        if (status != HPRQP_STATUS_OK) goto restore_state;
    }
    status = map_cuda_status(cudaDeviceSynchronize());
    stop = std::chrono::steady_clock::now();
    if (status != HPRQP_STATUS_OK) goto restore_state;
    *elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count() *
        (40.0 / (double)timing_repetitions);

restore_state:
    {
        HPRQPStatus restore_status =
            gpu_workspace_copy_parallel_benchmark_state(workspace,
                                                        saved_state);
        if (status == HPRQP_STATUS_OK &&
            restore_status != HPRQP_STATUS_OK) {
            status = restore_status;
        }
    }

restore_configuration:
    workspace->spmv_mode_q = saved_mode_q;
    workspace->spmv_mode_a = saved_mode_a;
    workspace->custom_spmv_q_kernel = saved_custom_q;
    workspace->custom_spmv_a_kernel = saved_custom_a;
    workspace->parallel_fuse_q = saved_fuse_q;
    workspace->parallel_fuse_at = saved_fuse_at;
    workspace->parallel_fuse_a = saved_fuse_a;
    workspace->to_check = saved_to_check;
    return status;
}

static HPRQPStatus benchmark_parallel_persistent_candidate(
    double *elapsed_ms,
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *model,
    const HPRQPGPUWorkspace *saved_state,
    unsigned int threads,
    unsigned int blocks,
    HPRQPGPUParallelPersistentKernel kernel_mode)
{
    static const size_t warmup_repetitions = 8U;
    static const size_t timing_repetitions = 100U;
    const int saved_to_check = workspace->to_check;
    HPRQPStatus status;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point stop;

    if (elapsed_ms == NULL || workspace == NULL || model == NULL ||
        saved_state == NULL || threads == 0U || blocks == 0U) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_workspace_copy_parallel_benchmark_state(workspace,
                                                         saved_state);
    if (status != HPRQP_STATUS_OK) goto restore_configuration;
    workspace->to_check = 0;
    status = gpu_iteration_batch_parallel_config(workspace,
                                                 model,
                                                 0,
                                                 warmup_repetitions,
                                                 threads,
                                                 blocks,
                                                 kernel_mode,
                                                 0);
    if (status != HPRQP_STATUS_OK) goto restore_state;
    status = map_cuda_status(cudaDeviceSynchronize());
    if (status != HPRQP_STATUS_OK) goto restore_state;

    status = gpu_workspace_copy_parallel_benchmark_state(workspace,
                                                         saved_state);
    if (status != HPRQP_STATUS_OK) goto restore_configuration;
    workspace->to_check = 0;
    status = map_cuda_status(cudaDeviceSynchronize());
    if (status != HPRQP_STATUS_OK) goto restore_state;
    start = std::chrono::steady_clock::now();
    status = gpu_iteration_batch_parallel_config(workspace,
                                                 model,
                                                 0,
                                                 timing_repetitions,
                                                 threads,
                                                 blocks,
                                                 kernel_mode,
                                                 0);
    if (status != HPRQP_STATUS_OK) goto restore_state;
    status = map_cuda_status(cudaDeviceSynchronize());
    stop = std::chrono::steady_clock::now();
    if (status != HPRQP_STATUS_OK) goto restore_state;
    *elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count() *
        (40.0 / (double)timing_repetitions);

restore_state:
    {
        HPRQPStatus restore_status =
            gpu_workspace_copy_parallel_benchmark_state(workspace,
                                                        saved_state);
        if (status == HPRQP_STATUS_OK &&
            restore_status != HPRQP_STATUS_OK) {
            status = restore_status;
        }
    }

restore_configuration:
    workspace->to_check = saved_to_check;
    return status;
}

static const char *custom_spmv_kernel_name_local(
    HPRQPGPUCustomSpmvKernel kernel)
{
    return kernel == HPRQP_GPU_CUSTOM_SPMV_WARP ? "warp" : "row";
}

static const char *persistent_kernel_name_local(
    HPRQPGPUParallelPersistentKernel kernel)
{
    switch (kernel) {
        case HPRQP_GPU_PARALLEL_PERSISTENT_ROW_PIPELINE:
            return "row-pipeline";
        case HPRQP_GPU_PARALLEL_PERSISTENT_GROUP:
            return "warp-group";
        case HPRQP_GPU_PARALLEL_PERSISTENT_ROW:
        default:
            return "row";
    }
}

static HPRQPStatus gpu_workspace_select_parallel_kernels(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *model,
    const HPRQPParams *params)
{
    HPRQPParallelQKernelCandidate q_candidates[4];
    HPRQPParallelAKernelCandidate a_candidates[6];
    HPRQPGPUWorkspace saved_state;
    HPRQPParallelQKernelCandidate best_q;
    HPRQPParallelAKernelCandidate best_a;
    size_t q_count = 0;
    size_t a_count = 0;
    size_t q_index;
    size_t a_index;
    double elapsed_ms;
    double best_ms = DBL_MAX;
    double best_persistent_ms = DBL_MAX;
    unsigned int best_persistent_threads = 0U;
    unsigned int best_persistent_blocks = 0U;
    unsigned int best_persistent_q_threads = 32U;
    int best_persistent_use_col16 = 0;
    HPRQPGPUParallelPersistentKernel best_persistent_kernel =
        HPRQP_GPU_PARALLEL_PERSISTENT_ROW;
    const char *persistent_env;
    int persistent_disabled;
    int persistent_forced;
    int persistent_allowed;
    int persistent_row_compatible;
    int persistent_group_compatible;
    int analytical_sparse_row_path = 0;
    int a_dominant_row6 = 0;
    int at_dominant_row6 = 0;
    int a_dominant_row10 = 0;
    int all_al_positive_zero = 0;
    int all_au_positive_zero = 0;
    unsigned int analytical_threads = 0U;
    unsigned int analytical_blocks = 0U;
    size_t analytical_resident_threads = 0U;
    HPRQPStatus status;

    status = gpu_parallel_prepare_q_nonempty_rows(workspace, model);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (model->p > 0U) {
        const int short_a_rows =
            model->A.nrow > 0U && model->AT.nrow > 0U &&
            model->A.nnz <= 4U * model->A.nrow &&
            model->AT.nnz <= 4U * model->AT.nrow;
        /*
         * The QP autotuner snapshots only the ordinary (x,y,w) state.  A QCQP
         * candidate run would therefore alter the cone-dual trajectory before
         * iteration zero.  Use a deterministic structure-based choice until a
         * full QCQP snapshot tuner is introduced.
         */
        workspace->spmv_mode_q =
            params->spmv_mode_q == HPRQP_SPMV_MODE_AUTO
                ? (workspace->Q_is_diag ? HPRQP_SPMV_MODE_CUSTOMIZED
                                        : HPRQP_SPMV_MODE_CUSPARSE)
                : params->spmv_mode_q;
        workspace->custom_spmv_q_kernel =
            workspace->Q_is_diag ? HPRQP_GPU_CUSTOM_SPMV_ROW
                                 : HPRQP_GPU_CUSTOM_SPMV_WARP;
        workspace->spmv_mode_a =
            params->spmv_mode_a == HPRQP_SPMV_MODE_AUTO
                ? (short_a_rows ? HPRQP_SPMV_MODE_CUSTOMIZED
                                : HPRQP_SPMV_MODE_CUSPARSE)
                : params->spmv_mode_a;
        workspace->custom_spmv_a_kernel =
            short_a_rows ? HPRQP_GPU_CUSTOM_SPMV_ROW
                         : HPRQP_GPU_CUSTOM_SPMV_WARP;
        workspace->parallel_fuse_q = 0;
        workspace->parallel_fuse_at = 0;
        workspace->parallel_fuse_a = 0;
        workspace->parallel_persistent_enabled = 0;
        if (params->verbose) {
            std::printf(
                "  QCQP kernel selector: Q=%s/%s, A=%s/%s, "
                "state-preserving structural choice "
                "(short A rows=%s, exact diagonal Q=%s)\n",
                spmv_mode_name_local(workspace->spmv_mode_q),
                custom_spmv_kernel_name_local(
                    workspace->custom_spmv_q_kernel),
                spmv_mode_name_local(workspace->spmv_mode_a),
                custom_spmv_kernel_name_local(
                    workspace->custom_spmv_a_kernel),
                short_a_rows ? "yes" : "no",
                workspace->parallel_q_exact_single_diagonal ? "yes" : "no");
        }
        return HPRQP_STATUS_OK;
    }

    workspace->parallel_persistent_fixed_row_length = 0U;

    status = gpu_parallel_vector_all_positive_zero(
        &model->l,
        &workspace->parallel_x_lower_positive_zero);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_parallel_vector_all_positive_zero(
        &model->AL,
        &all_al_positive_zero);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_parallel_vector_all_positive_zero(
        &model->AU,
        &all_au_positive_zero);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    workspace->parallel_constraints_positive_zero_equality =
        !workspace->empty_A && all_al_positive_zero &&
        all_au_positive_zero;
    {
        const char *projection_env =
            std::getenv("HPRQP_GPU_PROJECTION_SPECIALIZE");
        const int projection_disabled =
            projection_env != NULL &&
            (std::strcmp(projection_env, "0") == 0 ||
             std::strcmp(projection_env, "off") == 0);

        if (projection_disabled) {
            workspace->parallel_x_lower_positive_zero = 0;
            workspace->parallel_constraints_positive_zero_equality = 0;
        }
    }

    persistent_env = std::getenv("HPRQP_GPU_PARALLEL_BATCH");
    persistent_disabled =
        persistent_env != NULL &&
        (std::strcmp(persistent_env, "0") == 0 ||
         std::strcmp(persistent_env, "off") == 0);
    persistent_forced =
        persistent_env != NULL &&
        (std::strcmp(persistent_env, "1") == 0 ||
         std::strcmp(persistent_env, "on") == 0 ||
         std::strcmp(persistent_env, "force") == 0);
    persistent_allowed =
        !persistent_disabled &&
        (persistent_forced ||
         (params->spmv_mode_q != HPRQP_SPMV_MODE_CUSPARSE &&
          params->spmv_mode_a != HPRQP_SPMV_MODE_CUSPARSE));

    if (persistent_allowed && !workspace->empty_A &&
        workspace->Q_is_diag && model->Q.nnz > 0U &&
        model->A.nrow > 0U && model->A.ncol > 0U) {
        static const unsigned int analytical_thread_candidates[] = {
            128U, 256U, 512U};
        const size_t average_a_row_nnz =
            model->A.nnz / model->A.nrow +
            (model->A.nnz % model->A.nrow != 0U ? 1U : 0U);
        const size_t average_a_col_nnz =
            model->A.nnz / model->A.ncol +
            (model->A.nnz % model->A.ncol != 0U ? 1U : 0U);

        /*
         * A diagonal Q can be fused exactly.  When both average A row and
         * column lengths fit in one warp, the fused warp A/AT path is the
         * structural candidate.  CUDA occupancy then supplies a portable
         * cooperative grid for the current GPU (A100, H100, B200, MIG, ...).
         */
        if (average_a_row_nnz <= 32U && average_a_col_nnz <= 32U) {
            status = gpu_parallel_csr_has_dominant_row_length(
                &model->A,
                6,
                &a_dominant_row6);
            if (status != HPRQP_STATUS_OK) {
                return status;
            }
            status = gpu_parallel_csr_has_dominant_row_length(
                &model->AT,
                6,
                &at_dominant_row6);
            if (status != HPRQP_STATUS_OK) {
                return status;
            }
            if (a_dominant_row6 && at_dominant_row6) {
                workspace->parallel_persistent_fixed_row_length = 6U;
            } else {
                status = gpu_parallel_csr_has_dominant_row_length(
                    &model->A,
                    10,
                    &a_dominant_row10);
                if (status != HPRQP_STATUS_OK) {
                    return status;
                }
                if (a_dominant_row10 && model->A.ncol >= 65536U) {
                    workspace->parallel_persistent_fixed_row_length = 10U;
                }
            }

            for (size_t thread_index = 0U;
                 thread_index < sizeof(analytical_thread_candidates) /
                                        sizeof(analytical_thread_candidates[0]);
                 ++thread_index) {
                const unsigned int threads =
                    analytical_thread_candidates[thread_index];
                unsigned int maximum_blocks = 0U;
                const size_t row_count =
                    workspace->n > workspace->m
                        ? workspace->n
                        : workspace->m;
                const size_t natural_blocks =
                    row_count / threads +
                    (row_count % threads != 0U ? 1U : 0U);
                unsigned int blocks;
                size_t resident_threads;

                if (workspace->parallel_persistent_fixed_row_length == 6U &&
                    threads != 256U) {
                    continue;
                }

                status = gpu_parallel_persistent_config_for_threads(
                    threads,
                    0U,
                    &maximum_blocks,
                    HPRQP_GPU_PARALLEL_PERSISTENT_ROW,
                    0U,
                    workspace->parallel_persistent_fixed_row_length);
                if (status != HPRQP_STATUS_OK) {
                    return status;
                }
                if (maximum_blocks == 0U ||
                    natural_blocks < (size_t)maximum_blocks) {
                    continue;
                }
                blocks = maximum_blocks;
                resident_threads = (size_t)threads * (size_t)blocks;
                if (resident_threads > analytical_resident_threads ||
                    (resident_threads == analytical_resident_threads &&
                     threads > analytical_threads)) {
                    analytical_threads = threads;
                    analytical_blocks = blocks;
                    analytical_resident_threads = resident_threads;
                }
            }
            analytical_sparse_row_path = analytical_blocks > 0U;
        }
    }

    if (!analytical_sparse_row_path) {
        workspace->parallel_persistent_fixed_row_length = 0U;
    }

    if (analytical_sparse_row_path) {
        q_candidates[q_count++] = {
            HPRQP_SPMV_MODE_CUSTOMIZED,
            HPRQP_GPU_CUSTOM_SPMV_ROW,
            1};
    } else if (params->spmv_mode_q != HPRQP_SPMV_MODE_CUSTOMIZED) {
        q_candidates[q_count++] = {
            HPRQP_SPMV_MODE_CUSPARSE,
            HPRQP_GPU_CUSTOM_SPMV_WARP,
            0};
    }
    if (!analytical_sparse_row_path &&
        params->spmv_mode_q != HPRQP_SPMV_MODE_CUSPARSE) {
        q_candidates[q_count++] = {
            HPRQP_SPMV_MODE_CUSTOMIZED,
            HPRQP_GPU_CUSTOM_SPMV_ROW,
            0};
        /* In-place Q fusion is race-free only when each row reads its own w. */
        if (model->Q.nnz > 0U && workspace->Q_is_diag) {
            q_candidates[q_count++] = {
                HPRQP_SPMV_MODE_CUSTOMIZED,
                HPRQP_GPU_CUSTOM_SPMV_ROW,
                1};
        }
        q_candidates[q_count++] = {
            HPRQP_SPMV_MODE_CUSTOMIZED,
            HPRQP_GPU_CUSTOM_SPMV_WARP,
            0};
    }

    if (analytical_sparse_row_path) {
        a_candidates[a_count++] = {
            HPRQP_SPMV_MODE_CUSTOMIZED,
            HPRQP_GPU_CUSTOM_SPMV_WARP,
            1,
            1};
    } else if (workspace->empty_A) {
        a_candidates[a_count++] = {
            HPRQP_SPMV_MODE_CUSPARSE,
            HPRQP_GPU_CUSTOM_SPMV_WARP,
            0,
            0};
    } else {
        if (params->spmv_mode_a != HPRQP_SPMV_MODE_CUSTOMIZED) {
            a_candidates[a_count++] = {
                HPRQP_SPMV_MODE_CUSPARSE,
                HPRQP_GPU_CUSTOM_SPMV_WARP,
                0,
                0};
        }
        if (params->spmv_mode_a != HPRQP_SPMV_MODE_CUSPARSE) {
            a_candidates[a_count++] = {
                HPRQP_SPMV_MODE_CUSTOMIZED,
                HPRQP_GPU_CUSTOM_SPMV_ROW,
                0,
                0};
            a_candidates[a_count++] = {
                HPRQP_SPMV_MODE_CUSTOMIZED,
                HPRQP_GPU_CUSTOM_SPMV_WARP,
                0,
                0};
            a_candidates[a_count++] = {
                HPRQP_SPMV_MODE_CUSTOMIZED,
                HPRQP_GPU_CUSTOM_SPMV_WARP,
                1,
                0};
            a_candidates[a_count++] = {
                HPRQP_SPMV_MODE_CUSTOMIZED,
                HPRQP_GPU_CUSTOM_SPMV_WARP,
                0,
                1};
            a_candidates[a_count++] = {
                HPRQP_SPMV_MODE_CUSTOMIZED,
                HPRQP_GPU_CUSTOM_SPMV_WARP,
                1,
                1};
        }
    }
    if (q_count == 0 || a_count == 0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    status = gpu_parallel_prepare_col16(workspace, model);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (analytical_sparse_row_path) {
        workspace->spmv_mode_q = q_candidates[0].mode;
        workspace->spmv_mode_a = a_candidates[0].mode;
        workspace->custom_spmv_q_kernel = q_candidates[0].custom_kernel;
        workspace->custom_spmv_a_kernel = a_candidates[0].custom_kernel;
        workspace->parallel_fuse_q = q_candidates[0].fuse_q;
        workspace->parallel_fuse_at = a_candidates[0].fuse_at;
        workspace->parallel_fuse_a = a_candidates[0].fuse_a;
        workspace->parallel_persistent_enabled = 1;
        workspace->parallel_persistent_threads = analytical_threads;
        workspace->parallel_persistent_blocks = analytical_blocks;
        {
            const char *delta_env =
                std::getenv("HPRQP_GPU_ROW_DELTA16");
            const int delta_disabled =
                delta_env != NULL &&
                (std::strcmp(delta_env, "0") == 0 ||
                 std::strcmp(delta_env, "off") == 0);

            workspace->parallel_persistent_use_col16 =
                workspace->parallel_col16_available
                    ? 1
                    : (workspace->parallel_col16_delta_available &&
                               !delta_disabled
                           ? 2
                           : 0);
        }
        workspace->parallel_persistent_kernel =
            HPRQP_GPU_PARALLEL_PERSISTENT_ROW;
        workspace->parallel_persistent_q_group_width = 32U;
        if (params->verbose) {
            std::printf(
                "  Parallel kernel selector: analytical sparse row, "
                "Q=customized/row, A=customized/warp, "
                "fuse_Q=1, fuse_AT=1, fuse_A=1\n");
            std::printf(
                "  Parallel batch selector: occupancy formula, "
                "kernel=row, col16=%d, q_threads=32, threads=%u, "
                "blocks=%u, resident_threads=%zu, fixed_row=%u\n",
                workspace->parallel_persistent_use_col16,
                analytical_threads,
                analytical_blocks,
                analytical_resident_threads,
                workspace->parallel_persistent_fixed_row_length);
        }
        return HPRQP_STATUS_OK;
    }

    gpu_workspace_init(&saved_state);
    status = gpu_workspace_allocate(&saved_state, model, params);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_workspace_copy_parallel_benchmark_state(&saved_state,
                                                         workspace);
    if (status != HPRQP_STATUS_OK) {
        gpu_workspace_free(&saved_state);
        return status;
    }

    best_q = q_candidates[0];
    best_a = a_candidates[0];
    for (q_index = 0; q_index < q_count; ++q_index) {
        for (a_index = 0; a_index < a_count; ++a_index) {
            status = benchmark_parallel_kernel_candidate(
                &elapsed_ms,
                workspace,
                model,
                &saved_state,
                &q_candidates[q_index],
                &a_candidates[a_index]);
            if (status != HPRQP_STATUS_OK) {
                gpu_workspace_free(&saved_state);
                return status;
            }
            if (elapsed_ms < best_ms) {
                best_ms = elapsed_ms;
                best_q = q_candidates[q_index];
                best_a = a_candidates[a_index];
            }
        }
    }

    persistent_row_compatible =
        best_q.mode == HPRQP_SPMV_MODE_CUSTOMIZED &&
        best_q.custom_kernel == HPRQP_GPU_CUSTOM_SPMV_ROW &&
        (workspace->empty_A || (best_a.fuse_at && best_a.fuse_a));
    persistent_group_compatible =
        best_q.mode == HPRQP_SPMV_MODE_CUSTOMIZED &&
        best_q.custom_kernel == HPRQP_GPU_CUSTOM_SPMV_WARP;
    if (persistent_allowed && persistent_row_compatible) {
        static const unsigned int thread_candidates[] = {128U, 256U, 512U};

        /*
         * The pipeline can only hide useful Q work behind the A phases.
         * A diagonal or very light Q has too little independent work to
         * repay its shared-memory and synchronization overhead.
         */
        const size_t pipeline_q_work_floor =
            model->A.nnz / 8U + (model->A.nnz % 8U != 0U ? 1U : 0U);
        const int pipeline_has_balanced_work =
            !workspace->empty_A &&
            !workspace->Q_is_diag &&
            model->A.nnz > 0U &&
            model->Q.nnz > 0U &&
            model->Q.nnz >= pipeline_q_work_floor;
        const int maximum_pipeline_mode =
            pipeline_has_balanced_work ? 1 : 0;

        if (params->verbose > 1) {
            std::printf(
                "    row-batch structural shortlist: n=%zu, m=%zu, "
                "nnz_Q=%zu, nnz_A=%zu, Q_diag=%d, pipeline=%d\n",
                workspace->n,
                workspace->m,
                model->Q.nnz,
                model->A.nnz,
                workspace->Q_is_diag,
                maximum_pipeline_mode);
        }

        for (int pipeline_mode = 0;
             pipeline_mode <= maximum_pipeline_mode;
             ++pipeline_mode) {
        const HPRQPGPUParallelPersistentKernel kernel_mode =
            pipeline_mode
                ? HPRQP_GPU_PARALLEL_PERSISTENT_ROW_PIPELINE
                : HPRQP_GPU_PARALLEL_PERSISTENT_ROW;
        for (int col16_mode = 0;
             col16_mode <= (workspace->parallel_col16_available ? 1 : 0);
             ++col16_mode) {
        workspace->parallel_persistent_use_col16 = col16_mode;
        for (size_t thread_index = 0;
             thread_index < sizeof(thread_candidates) /
                                    sizeof(thread_candidates[0]);
             ++thread_index) {
            unsigned int maximum_blocks = 0U;
            unsigned int block_candidates[13];
            unsigned int seen_blocks[13] = {0U};
            size_t block_count = 0U;
            size_t seen_count = 0U;
            const size_t row_count =
                workspace->n > workspace->m
                    ? workspace->n
                    : workspace->m;
            const size_t natural_block_count =
                (row_count + thread_candidates[thread_index] - 1U) /
                thread_candidates[thread_index];
            const size_t phase_row_count =
                workspace->n + (workspace->empty_A ? 0U : workspace->m);
            const size_t phase_block_count =
                (phase_row_count + thread_candidates[thread_index] - 1U) /
                thread_candidates[thread_index];

            status = gpu_parallel_persistent_config_for_threads(
                thread_candidates[thread_index],
                0U,
                &maximum_blocks,
                kernel_mode,
                0U);
            if (status != HPRQP_STATUS_OK) {
                gpu_workspace_free(&saved_state);
                return status;
            }
            if (maximum_blocks == 0U) {
                continue;
            }
            const int occupancy_shortlist =
                !pipeline_mode &&
                !workspace->empty_A &&
                !pipeline_has_balanced_work &&
                natural_block_count >= (size_t)maximum_blocks;

            /*
             * For a large A-dominated row kernel, every candidate below is
             * just a different decomposition of the same full resident
             * thread population.  NVIDIA's occupancy guidance makes the
             * maximum cooperative grid the analytical block candidate.  A
             * 256/512-thread micro-selector retains one measured choice for
             * register/ILP effects while avoiding the redundant 128-thread
             * geometry and quarter/half-occupancy launches.
             */
            if (occupancy_shortlist &&
                thread_candidates[thread_index] == 128U) {
                continue;
            }
            /*
             * Keep the fine low-block sweep for genuinely small models and
             * for the shared-memory pipeline.  Once the row population can
             * saturate at least a quarter of the cooperative grid, only
             * natural and quarter/half/full-occupancy geometries are useful;
             * 1--64 block launches merely leave most SMs idle.
             */
            if (!occupancy_shortlist &&
                (pipeline_mode ||
                 natural_block_count <= (size_t)maximum_blocks / 4U)) {
                block_candidates[block_count++] = 1U;
                block_candidates[block_count++] = 2U;
                block_candidates[block_count++] = 4U;
                block_candidates[block_count++] = 8U;
                block_candidates[block_count++] = 16U;
                block_candidates[block_count++] = 32U;
                block_candidates[block_count++] = 64U;
            }
            if (!occupancy_shortlist) {
                block_candidates[block_count++] =
                    natural_block_count > maximum_blocks
                        ? maximum_blocks
                        : (unsigned int)natural_block_count;
                block_candidates[block_count++] =
                    natural_block_count > (size_t)maximum_blocks / 2U
                        ? maximum_blocks
                        : (unsigned int)(2U * natural_block_count);
                block_candidates[block_count++] =
                    phase_block_count > maximum_blocks
                        ? maximum_blocks
                        : (unsigned int)phase_block_count;
                block_candidates[block_count++] = maximum_blocks / 4U;
                block_candidates[block_count++] = maximum_blocks / 2U;
            }
            block_candidates[block_count++] = maximum_blocks;

            for (size_t block_index = 0;
                 block_index < block_count;
                 ++block_index) {
                const unsigned int blocks = block_candidates[block_index];
                int duplicate = 0;

                if (blocks == 0U || blocks > maximum_blocks) {
                    continue;
                }
                for (size_t seen_index = 0;
                     seen_index < seen_count;
                     ++seen_index) {
                    if (seen_blocks[seen_index] == blocks) {
                        duplicate = 1;
                        break;
                    }
                }
                if (duplicate) {
                    continue;
                }
                seen_blocks[seen_count++] = blocks;
                {
                    unsigned int q_thread_candidates[6] = {32U};
                    unsigned int seen_q_threads[6] = {0U};
                    size_t q_thread_count = 1U;
                    size_t seen_q_count = 0U;

                    if (pipeline_mode) {
                        const size_t rows_per_block =
                            workspace->n / (size_t)blocks +
                            (workspace->n % (size_t)blocks != 0U
                                 ? 1U
                                 : 0U);
                        const size_t dynamic_shared_bytes =
                            rows_per_block * sizeof(double);
                        unsigned int shared_limited_blocks = 0U;

                        status =
                            gpu_parallel_persistent_config_for_threads(
                                thread_candidates[thread_index],
                                0U,
                                &shared_limited_blocks,
                                HPRQP_GPU_PARALLEL_PERSISTENT_ROW_PIPELINE,
                                dynamic_shared_bytes);
                        if (status != HPRQP_STATUS_OK) {
                            gpu_workspace_free(&saved_state);
                            return status;
                        }
                        if (blocks > shared_limited_blocks) {
                            continue;
                        }
                        q_thread_candidates[1] =
                            thread_candidates[thread_index] / 4U;
                        q_thread_candidates[2] =
                            thread_candidates[thread_index] / 2U;
                        q_thread_candidates[3] =
                            (5U * thread_candidates[thread_index] / 8U) &
                            ~31U;
                        q_thread_candidates[4] =
                            3U * thread_candidates[thread_index] / 4U;
                        q_thread_candidates[5] =
                            thread_candidates[thread_index] - 32U;
                        q_thread_count = 6U;
                    }

                    for (size_t q_thread_index = 0U;
                         q_thread_index < q_thread_count;
                         ++q_thread_index) {
                        const unsigned int q_threads =
                            q_thread_candidates[q_thread_index];
                        int duplicate_q_threads = 0;

                        if (q_threads < 32U ||
                            q_threads >
                                thread_candidates[thread_index] ||
                            (q_threads & 31U) != 0U ||
                            (pipeline_mode &&
                             q_threads ==
                                 thread_candidates[thread_index])) {
                            continue;
                        }
                        for (size_t seen_q_index = 0U;
                             seen_q_index < seen_q_count;
                             ++seen_q_index) {
                            if (seen_q_threads[seen_q_index] ==
                                q_threads) {
                                duplicate_q_threads = 1;
                                break;
                            }
                        }
                        if (duplicate_q_threads) {
                            continue;
                        }
                        seen_q_threads[seen_q_count++] = q_threads;
                        workspace->parallel_persistent_q_group_width =
                            q_threads;
                        status = benchmark_parallel_persistent_candidate(
                            &elapsed_ms,
                            workspace,
                            model,
                            &saved_state,
                            thread_candidates[thread_index],
                            blocks,
                            kernel_mode);
                        if (status != HPRQP_STATUS_OK) {
                            gpu_workspace_free(&saved_state);
                            return status;
                        }
                        if (params->verbose > 1) {
                            std::printf(
                                "    row-batch candidate: pipeline=%d, "
                                "col16=%d, q_threads=%u, threads=%u, "
                                "blocks=%u, sample=%.6f ms\n",
                                pipeline_mode,
                                col16_mode,
                                q_threads,
                                thread_candidates[thread_index],
                                blocks,
                                elapsed_ms);
                        }
                        {
                            const int comparing_row_geometries =
                                best_persistent_blocks > 0U &&
                                kernel_mode ==
                                    HPRQP_GPU_PARALLEL_PERSISTENT_ROW &&
                                best_persistent_kernel ==
                                    HPRQP_GPU_PARALLEL_PERSISTENT_ROW;
                            const int row_geometry_tie =
                                comparing_row_geometries &&
                                elapsed_ms <= best_persistent_ms * 1.005 &&
                                thread_candidates[thread_index] >
                                    best_persistent_threads;
                            const int materially_faster =
                                best_persistent_blocks == 0U ||
                                elapsed_ms < best_persistent_ms * 0.995;
                            const int strictly_faster =
                                elapsed_ms < best_persistent_ms;

                            /*
                             * Full-occupancy row geometries often differ by
                             * less than timer noise because they expose the
                             * same resident-thread count.  Use the larger
                             * block as a deterministic tie-break; require a
                             * 0.5% win to replace it with a smaller block.
                             */
                            if ((comparing_row_geometries &&
                                 !materially_faster &&
                                 !row_geometry_tie) ||
                                (!comparing_row_geometries &&
                                 !strictly_faster)) {
                                continue;
                            }
                            best_persistent_ms = elapsed_ms;
                            best_persistent_threads =
                                thread_candidates[thread_index];
                            best_persistent_blocks = blocks;
                            best_persistent_q_threads = q_threads;
                            best_persistent_use_col16 = col16_mode;
                            best_persistent_kernel = kernel_mode;
                        }
                    }
                }
            }
        }
        }
        }

    }
    if (persistent_allowed &&
        (persistent_group_compatible || persistent_forced)) {
        static const unsigned int thread_candidates[] = {128U, 256U, 512U};
        static const unsigned int q_group_candidates[] = {
            32U, 64U, 128U, 256U};
        double best_group_ms = DBL_MAX;
        unsigned int best_group_threads = 0U;
        unsigned int best_group_blocks = 0U;
        unsigned int best_group_q_width = 32U;
        int best_group_use_col16 = 0;

        /*
         * The ordinary warp SpMV and this persistent kernel use the same
         * structural regime: one or more complete warps cooperate on a Q
         * row.  Keep its search separate from the exact row kernels so a row
         * candidate cannot accidentally supply the group launch geometry.
         */
        for (int col16_mode = 0;
             col16_mode <= (workspace->parallel_col16_available ? 1 : 0);
             ++col16_mode) {
            workspace->parallel_persistent_use_col16 = col16_mode;
            workspace->parallel_persistent_q_group_width = 32U;
            for (size_t thread_index = 0;
                 thread_index < sizeof(thread_candidates) /
                                    sizeof(thread_candidates[0]);
                 ++thread_index) {
                unsigned int maximum_blocks = 0U;
                unsigned int block_candidates[10];
                unsigned int seen_blocks[10] = {0U};
                size_t block_count = 0U;
                size_t seen_count = 0U;

                status = gpu_parallel_persistent_config_for_threads(
                    thread_candidates[thread_index],
                    0U,
                    &maximum_blocks,
                    HPRQP_GPU_PARALLEL_PERSISTENT_GROUP,
                    0U);
                if (status != HPRQP_STATUS_OK) {
                    gpu_workspace_free(&saved_state);
                    return status;
                }
                if (maximum_blocks == 0U) {
                    continue;
                }
                block_candidates[block_count++] = 1U;
                block_candidates[block_count++] = 2U;
                block_candidates[block_count++] = 4U;
                block_candidates[block_count++] = 8U;
                block_candidates[block_count++] = 16U;
                block_candidates[block_count++] = 32U;
                block_candidates[block_count++] = 64U;
                block_candidates[block_count++] = maximum_blocks / 4U;
                block_candidates[block_count++] = maximum_blocks / 2U;
                block_candidates[block_count++] = maximum_blocks;

                for (size_t block_index = 0;
                     block_index < block_count;
                     ++block_index) {
                    const unsigned int blocks =
                        block_candidates[block_index];
                    int duplicate = 0;

                    if (blocks == 0U || blocks > maximum_blocks) {
                        continue;
                    }
                    for (size_t seen_index = 0;
                         seen_index < seen_count;
                         ++seen_index) {
                        if (seen_blocks[seen_index] == blocks) {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (duplicate) {
                        continue;
                    }
                    seen_blocks[seen_count++] = blocks;
                    status = benchmark_parallel_persistent_candidate(
                        &elapsed_ms,
                        workspace,
                        model,
                        &saved_state,
                        thread_candidates[thread_index],
                        blocks,
                        HPRQP_GPU_PARALLEL_PERSISTENT_GROUP);
                    if (status != HPRQP_STATUS_OK) {
                        gpu_workspace_free(&saved_state);
                        return status;
                    }
                    if (params->verbose > 1) {
                        std::printf(
                            "    warp-group candidate: col16=%d, "
                            "q_group=32, threads=%u, blocks=%u, "
                            "sample=%.6f ms\n",
                            col16_mode,
                            thread_candidates[thread_index],
                            blocks,
                            elapsed_ms);
                    }
                    if (elapsed_ms < best_group_ms) {
                        best_group_ms = elapsed_ms;
                        best_group_threads =
                            thread_candidates[thread_index];
                        best_group_blocks = blocks;
                        best_group_q_width = 32U;
                        best_group_use_col16 = col16_mode;
                    }
                }
            }
        }

        if (best_group_blocks > 0U) {
            workspace->parallel_persistent_use_col16 =
                best_group_use_col16;
            for (size_t q_group_index = 1U;
                 q_group_index < sizeof(q_group_candidates) /
                                     sizeof(q_group_candidates[0]);
                 ++q_group_index) {
                const unsigned int q_group_width =
                    q_group_candidates[q_group_index];

                if (q_group_width > best_group_threads ||
                    best_group_threads % q_group_width != 0U) {
                    continue;
                }
                workspace->parallel_persistent_q_group_width =
                    q_group_width;
                status = benchmark_parallel_persistent_candidate(
                    &elapsed_ms,
                    workspace,
                    model,
                    &saved_state,
                    best_group_threads,
                    best_group_blocks,
                    HPRQP_GPU_PARALLEL_PERSISTENT_GROUP);
                if (status != HPRQP_STATUS_OK) {
                    gpu_workspace_free(&saved_state);
                    return status;
                }
                if (params->verbose > 1) {
                    std::printf(
                        "    warp-group long-row candidate: col16=%d, "
                        "q_group=%u, threads=%u, blocks=%u, "
                        "sample=%.6f ms\n",
                        best_group_use_col16,
                        q_group_width,
                        best_group_threads,
                        best_group_blocks,
                        elapsed_ms);
                }
                if (elapsed_ms < best_group_ms) {
                    best_group_ms = elapsed_ms;
                    best_group_q_width = q_group_width;
                }
            }
        }

        if (best_group_ms < best_persistent_ms) {
            best_persistent_ms = best_group_ms;
            best_persistent_threads = best_group_threads;
            best_persistent_blocks = best_group_blocks;
            best_persistent_q_threads = best_group_q_width;
            best_persistent_use_col16 = best_group_use_col16;
            best_persistent_kernel =
                HPRQP_GPU_PARALLEL_PERSISTENT_GROUP;
        }
    }
    gpu_workspace_free(&saved_state);

    workspace->spmv_mode_q = best_q.mode;
    workspace->spmv_mode_a = best_a.mode;
    workspace->custom_spmv_q_kernel = best_q.custom_kernel;
    workspace->custom_spmv_a_kernel = best_a.custom_kernel;
    workspace->parallel_fuse_q = best_q.fuse_q;
    workspace->parallel_fuse_at = best_a.fuse_at;
    workspace->parallel_fuse_a = best_a.fuse_a;
    workspace->parallel_persistent_enabled =
        best_persistent_blocks > 0U &&
        (persistent_forced || best_persistent_ms < best_ms);
    workspace->parallel_persistent_threads = best_persistent_threads;
    workspace->parallel_persistent_blocks = best_persistent_blocks;
    workspace->parallel_persistent_use_col16 =
        best_persistent_use_col16;
    workspace->parallel_persistent_kernel =
        best_persistent_kernel;
    workspace->parallel_persistent_q_group_width =
        best_persistent_q_threads;
    if (params->verbose) {
        std::printf(
            "  Parallel kernel autotune: Q=%s/%s, A=%s/%s, "
            "fuse_Q=%d, fuse_AT=%d, fuse_A=%d, sample=%.6f ms\n",
            spmv_mode_name_local(best_q.mode),
            custom_spmv_kernel_name_local(best_q.custom_kernel),
            spmv_mode_name_local(best_a.mode),
            custom_spmv_kernel_name_local(best_a.custom_kernel),
            best_q.fuse_q,
            best_a.fuse_at,
            best_a.fuse_a,
            best_ms);
        if (best_persistent_blocks > 0U) {
            std::printf(
                "  Parallel batch autotune: kernel=%s, col16=%d, "
                "q_threads=%u, threads=%u, blocks=%u, "
                "sample=%.6f ms, selected=%d\n",
                persistent_kernel_name_local(best_persistent_kernel),
                best_persistent_use_col16,
                best_persistent_q_threads,
                best_persistent_threads,
                best_persistent_blocks,
                best_persistent_ms,
                workspace->parallel_persistent_enabled);
        }
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_workspace_select_spmv_modes(HPRQPGPUWorkspace *workspace,
                                            const HPRQPGPUProblem *model,
                                            const HPRQPParams *params)
{
    HPRQPStatus status;
    HPRQPGPUWorkspace benchmark_workspace;
    HPRQPSpmvMode selected_mode_q;
    HPRQPSpmvMode selected_mode_a;
    double q_cusparse_ms;
    double q_custom_ms;
    double a_cusparse_ms;
    double a_custom_ms;
    int benchmark_q;
    int benchmark_a;
    int need_benchmark;

    if (workspace == NULL || model == NULL || params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (params->method == HPRQP_METHOD_PARALLEL) {
        return gpu_workspace_select_parallel_kernels(workspace,
                                                     model,
                                                     params);
    }

    gpu_workspace_init(&benchmark_workspace);

    workspace->spmv_mode_q = params->spmv_mode_q == HPRQP_SPMV_MODE_AUTO ? HPRQP_SPMV_MODE_CUSPARSE : params->spmv_mode_q;
    workspace->spmv_mode_a = params->spmv_mode_a == HPRQP_SPMV_MODE_AUTO ? HPRQP_SPMV_MODE_CUSPARSE : params->spmv_mode_a;
    selected_mode_q = workspace->spmv_mode_q;
    selected_mode_a = workspace->spmv_mode_a;

    benchmark_q = params->spmv_mode_q == HPRQP_SPMV_MODE_AUTO && model->Q.nnz > 0;
    benchmark_a = !workspace->empty_A && params->spmv_mode_a == HPRQP_SPMV_MODE_AUTO;
    need_benchmark = benchmark_q || benchmark_a;
    if (need_benchmark) {
        status = gpu_workspace_allocate(&benchmark_workspace, model, params);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (benchmark_q) {
        status = benchmark_q_mode(&q_cusparse_ms, workspace, model, &benchmark_workspace, HPRQP_SPMV_MODE_CUSPARSE);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace_free(&benchmark_workspace);
            return status;
        }
        status = benchmark_q_mode(&q_custom_ms, workspace, model, &benchmark_workspace, HPRQP_SPMV_MODE_CUSTOMIZED);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace_free(&benchmark_workspace);
            return status;
        }
        selected_mode_q = q_cusparse_ms <= q_custom_ms ? HPRQP_SPMV_MODE_CUSPARSE : HPRQP_SPMV_MODE_CUSTOMIZED;
        workspace->spmv_mode_q = selected_mode_q;
        // if (params->verbose) {
        //     std::printf("    Q: CUSPARSE=%.3fms, customized=%.3fms -> %s\n",
        //                 q_cusparse_ms,
        //                 q_custom_ms,
        //                 spmv_mode_name_local(workspace->spmv_mode_q));
        // }
    }

    if (benchmark_a) {
        status = benchmark_a_mode(&a_cusparse_ms, workspace, model, &benchmark_workspace, HPRQP_SPMV_MODE_CUSPARSE);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace_free(&benchmark_workspace);
            return status;
        }
        status = benchmark_a_mode(&a_custom_ms, workspace, model, &benchmark_workspace, HPRQP_SPMV_MODE_CUSTOMIZED);
        if (status != HPRQP_STATUS_OK) {
            gpu_workspace_free(&benchmark_workspace);
            return status;
        }
        selected_mode_a = a_cusparse_ms <= a_custom_ms ? HPRQP_SPMV_MODE_CUSPARSE : HPRQP_SPMV_MODE_CUSTOMIZED;
        workspace->spmv_mode_a = selected_mode_a;
        // if (params->verbose) {
        //     std::printf("    A: CUSPARSE=%.3fms, customized=%.3fms -> %s\n",
        //                 a_cusparse_ms,
        //                 a_custom_ms,
        //                 spmv_mode_name_local(workspace->spmv_mode_a));
        // }
    }

    if (need_benchmark) {
        gpu_workspace_free(&benchmark_workspace);
        workspace->spmv_mode_q = selected_mode_q;
        workspace->spmv_mode_a = selected_mode_a;
    }

    // if (params->verbose) {
    //     std::printf("  Selected: spmv_mode_q=%s, spmv_mode_a=%s\n",
    //                 spmv_mode_name_local(workspace->spmv_mode_q),
    //                 spmv_mode_name_local(workspace->spmv_mode_a));
    // }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_iteration_step_parallel(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPGPUProblem *gpu_problem,
                                        size_t inner_iter)
{
    HPRQPStatus status;
    double halpern_fact1;
    double halpern_fact2;

    if (gpu_workspace == NULL || gpu_problem == NULL ||
        !gpu_workspace->allocated || !gpu_problem->uploaded) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (gpu_problem->p > 0) {
        return gpu_qcqp_iteration_step_parallel(gpu_workspace,
                                                gpu_problem,
                                                inner_iter);
    }
    halpern_fact1 = 1.0 / ((double)inner_iter + 2.0);
    halpern_fact2 = 1.0 - halpern_fact1;

    if (gpu_problem->Q.nnz > 0 &&
        !gpu_workspace->parallel_fuse_q) {
        status = gpu_spmv_apply_q_w(gpu_workspace, gpu_problem);
    } else if (gpu_problem->Q.nnz == 0) {
        status = map_cuda_status(cudaMemset(gpu_workspace->Qw.values,
                                            0,
                                            gpu_workspace->n * sizeof(double)));
    } else {
        status = HPRQP_STATUS_OK;
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (!gpu_workspace->empty_A && !gpu_workspace->parallel_fuse_at) {
        status = gpu_spmv_apply_at_y(gpu_workspace, gpu_problem);
    } else if (gpu_workspace->empty_A) {
        status = map_cuda_status(cudaMemset(gpu_workspace->ATy.values,
                                            0,
                                            gpu_workspace->n * sizeof(double)));
    } else {
        status = HPRQP_STATUS_OK;
    }
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (gpu_workspace->parallel_fuse_q) {
        gpu_parallel_update_xw_fused_q_row_kernel<<<
            gpu_workspace->blocks_n,
            gpu_workspace->threads_n>>>(
            gpu_workspace->dx.values,
            gpu_workspace->dw.values,
            gpu_workspace->x_bar.values,
            gpu_workspace->x_hat.values,
            gpu_workspace->z_bar.values,
            gpu_workspace->w_bar.values,
            gpu_workspace->x.values,
            gpu_workspace->w.values,
            gpu_workspace->last_x.values,
            gpu_workspace->last_w.values,
            gpu_workspace->Qw.values,
            gpu_problem->Q.row_ptr,
            gpu_problem->Q.col_ind,
            gpu_problem->Q.values,
            gpu_workspace->ATy.values,
            gpu_workspace->y.values,
            gpu_problem->AT.row_ptr,
            gpu_problem->AT.col_ind,
            gpu_problem->AT.values,
            gpu_problem->c.values,
            gpu_problem->l.values,
            gpu_problem->u.values,
            gpu_workspace->sigma,
            gpu_workspace->parallel_w_gamma,
            gpu_workspace->parallel_w_inv,
            halpern_fact1,
            halpern_fact2,
            (!gpu_workspace->empty_A &&
             gpu_workspace->parallel_fuse_at) ? 1 : 0,
            gpu_workspace->to_check,
            gpu_workspace->n);
    } else if (!gpu_workspace->empty_A &&
               gpu_workspace->parallel_fuse_at) {
        gpu_parallel_update_xw_fused_at_row_kernel<<<
            gpu_workspace->blocks_n,
            gpu_workspace->threads_n>>>(
            gpu_workspace->dx.values,
            gpu_workspace->dw.values,
            gpu_workspace->x_bar.values,
            gpu_workspace->x_hat.values,
            gpu_workspace->z_bar.values,
            gpu_workspace->w_bar.values,
            gpu_workspace->x.values,
            gpu_workspace->w.values,
            gpu_workspace->last_x.values,
            gpu_workspace->last_w.values,
            gpu_workspace->Qw.values,
            gpu_workspace->ATy.values,
            gpu_workspace->y.values,
            gpu_problem->AT.row_ptr,
            gpu_problem->AT.col_ind,
            gpu_problem->AT.values,
            gpu_problem->c.values,
            gpu_problem->l.values,
            gpu_problem->u.values,
            gpu_workspace->sigma,
            gpu_workspace->parallel_w_gamma,
            gpu_workspace->parallel_w_inv,
            halpern_fact1,
            halpern_fact2,
            gpu_workspace->to_check,
            gpu_workspace->n);
    } else {
        gpu_parallel_update_xw_kernel<<<gpu_workspace->blocks_n,
                                        gpu_workspace->threads_n>>>(
            gpu_workspace->dx.values,
            gpu_workspace->dw.values,
            gpu_workspace->x_bar.values,
            gpu_workspace->x_hat.values,
            gpu_workspace->z_bar.values,
            gpu_workspace->w_bar.values,
            gpu_workspace->x.values,
            gpu_workspace->w.values,
            gpu_workspace->last_x.values,
            gpu_workspace->last_w.values,
            gpu_workspace->Qw.values,
            gpu_workspace->ATy.values,
            gpu_problem->c.values,
            gpu_problem->l.values,
            gpu_problem->u.values,
            gpu_workspace->sigma,
            gpu_workspace->parallel_w_gamma,
            gpu_workspace->parallel_w_inv,
            halpern_fact1,
            halpern_fact2,
            gpu_workspace->to_check,
            gpu_workspace->n);
    }
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    if (!gpu_workspace->empty_A) {
        if (gpu_workspace->parallel_fuse_a) {
            status = HPRQP_STATUS_OK;
        } else {
            status = gpu_spmv_apply_a_x_hat(gpu_workspace, gpu_problem);
        }
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        if (gpu_workspace->parallel_fuse_a) {
            gpu_parallel_update_y_fused_a_row_kernel<<<
                gpu_workspace->blocks_m,
                gpu_workspace->threads_m>>>(
                gpu_workspace->dy.values,
                gpu_workspace->s.values,
                gpu_workspace->y_bar.values,
                gpu_workspace->y.values,
                gpu_workspace->last_y.values,
                gpu_workspace->Ax.values,
                gpu_workspace->x_hat.values,
                gpu_problem->A.row_ptr,
                gpu_problem->A.col_ind,
                gpu_problem->A.values,
                gpu_problem->AL.values,
                gpu_problem->AU.values,
                gpu_workspace->parallel_y_rho,
                gpu_workspace->parallel_y_inv,
                halpern_fact1,
                halpern_fact2,
                gpu_workspace->to_check,
                gpu_workspace->m);
        } else {
            gpu_parallel_update_y_kernel<<<gpu_workspace->blocks_m,
                                          gpu_workspace->threads_m>>>(
                gpu_workspace->dy.values,
                gpu_workspace->s.values,
                gpu_workspace->y_bar.values,
                gpu_workspace->y.values,
                gpu_workspace->last_y.values,
                gpu_workspace->Ax.values,
                gpu_problem->AL.values,
                gpu_problem->AU.values,
                gpu_workspace->parallel_y_rho,
                gpu_workspace->parallel_y_inv,
                halpern_fact1,
                halpern_fact2,
                gpu_workspace->to_check,
                gpu_workspace->m);
        }
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    if (gpu_workspace->to_check) {
        return gpu_parallel_finish_check(gpu_workspace, gpu_problem);
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_iteration_step(HPRQPGPUWorkspace *gpu_workspace,
                               const HPRQPGPUProblem *gpu_problem,
                               size_t inner_iter)
{
    double halpern_fact1;
    double halpern_fact2;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;

    if (gpu_problem != NULL && gpu_problem->p > 0 &&
        (gpu_workspace == NULL ||
         gpu_workspace->method != HPRQP_METHOD_PARALLEL)) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    if (gpu_workspace != NULL &&
        gpu_workspace->method == HPRQP_METHOD_PARALLEL) {
        return gpu_iteration_step_parallel(gpu_workspace,
                                           gpu_problem,
                                           inner_iter);
    }

    halpern_fact1 = 1.0 / ((double)inner_iter + 2.0);
    halpern_fact2 = 1.0 - halpern_fact1;
    threads_n = gpu_workspace->threads_n;
    blocks_n = gpu_workspace->blocks_n;
    threads_m = gpu_workspace->threads_m;
    blocks_m = gpu_workspace->blocks_m;

    if (gpu_workspace->noC) {
        (void)gpu_update_zxw1_noC_step(gpu_workspace,
                                       gpu_problem,
                                       halpern_fact1,
                                       halpern_fact2,
                                       threads_n,
                                       blocks_n);
    } else {
        (void)gpu_update_zxw1_step(gpu_workspace,
                                   gpu_problem,
                                   halpern_fact1,
                                   halpern_fact2,
                                   threads_n,
                                   blocks_n);
    }

    (void)gpu_update_y_step(gpu_workspace,
                            gpu_problem,
                            halpern_fact1,
                            halpern_fact2,
                            threads_m,
                            blocks_m);

    if (gpu_workspace->noC) {
        return HPRQP_STATUS_OK;
    }

    (void)gpu_update_w2_step(gpu_workspace,
                             gpu_problem,
                             halpern_fact1,
                             halpern_fact2,
                             threads_n,
                             blocks_n);

    return HPRQP_STATUS_OK;
}
