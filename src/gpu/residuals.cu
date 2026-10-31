#include "cuda_common.h"
#include "gpu/qcqp.h"

#include <stdlib.h>

enum {
    HPRQP_GPU_FUSED_CHECK_THREADS = 256,
    HPRQP_GPU_FUSED_CHECK_MAX_LENGTH = 32768
};

__global__ static void gpu_parallel_fused_residual_stats_kernel(
    double *output,
    double *rd,
    double *rp,
    double *scaled_qx_or_ax_abs,
    double *scaled_aty_abs,
    double *s,
    const double *x_bar,
    const double *y_bar,
    const double *z_bar,
    const double *qx,
    const double *aty_bar,
    const double *ax,
    const double *c,
    const double *l,
    const double *u,
    const double *al,
    const double *au,
    const double *col_norm,
    const double *row_norm,
    double c_scale,
    double b_scale,
    size_t n,
    size_t m,
    int include_bound)
{
    __shared__ double partial[10][HPRQP_GPU_FUSED_CHECK_THREADS];
    const unsigned int tid = threadIdx.x;
    double local_c_dot_x = 0.0;
    double local_x_dot_qx = 0.0;
    double local_z_dot_x = 0.0;
    double local_y_dot_s = 0.0;
    double local_max_rd = 0.0;
    double local_max_qx = 0.0;
    double local_max_aty = 0.0;
    double local_max_rp = 0.0;
    double local_max_ax = 0.0;
    double local_bound = 0.0;

    for (size_t i = (size_t)tid; i < n;
         i += (size_t)HPRQP_GPU_FUSED_CHECK_THREADS) {
        const double x_i = x_bar[i];
        const double qx_i = qx[i];
        const double aty_i = aty_bar[i];
        const double scale_fact = col_norm[i] * c_scale;
        const double rd_i =
            (qx_i + c[i] - aty_i - z_bar[i]) * scale_fact;
        const double scaled_qx_i = fabs(qx_i * scale_fact);
        const double scaled_aty_i = fabs(aty_i * scale_fact);

        rd[i] = rd_i;
        scaled_qx_or_ax_abs[i] = scaled_qx_i;
        scaled_aty_abs[i] = scaled_aty_i;
        local_c_dot_x += c[i] * x_i;
        local_x_dot_qx += x_i * qx_i;
        local_z_dot_x += z_bar[i] * x_i;
        local_max_rd = fmax(local_max_rd, fabs(rd_i));
        local_max_qx = fmax(local_max_qx, scaled_qx_i);
        local_max_aty = fmax(local_max_aty, scaled_aty_i);

        if (include_bound) {
            const double lower_violation = l[i] - x_i;
            const double upper_violation = x_i - u[i];
            double violation = 0.0;
            if (lower_violation > 0.0) {
                violation += lower_violation;
            }
            if (upper_violation > 0.0) {
                violation += upper_violation;
            }
            local_bound = fmax(
                local_bound,
                fabs(violation * (b_scale / col_norm[i])));
        }
    }

    for (size_t i = (size_t)tid; i < m;
         i += (size_t)HPRQP_GPU_FUSED_CHECK_THREADS) {
        const double ax_i = ax[i];
        const double projected =
            ax_i < al[i] ? al[i] : (ax_i > au[i] ? au[i] : ax_i);
        const double scale_fact = row_norm[i] * b_scale;
        const double rp_i = (projected - ax_i) * scale_fact;
        const double scaled_ax_i = fabs(ax_i * scale_fact);

        s[i] = projected;
        rp[i] = rp_i;
        scaled_qx_or_ax_abs[i] = scaled_ax_i;
        local_y_dot_s += y_bar[i] * projected;
        local_max_rp = fmax(local_max_rp, fabs(rp_i));
        local_max_ax = fmax(local_max_ax, scaled_ax_i);
    }

    partial[0][tid] = local_c_dot_x;
    partial[1][tid] = local_x_dot_qx;
    partial[2][tid] = local_z_dot_x;
    partial[3][tid] = local_y_dot_s;
    partial[4][tid] = local_max_rd;
    partial[5][tid] = local_max_qx;
    partial[6][tid] = local_max_aty;
    partial[7][tid] = local_max_rp;
    partial[8][tid] = local_max_ax;
    partial[9][tid] = local_bound;
    __syncthreads();

    for (unsigned int stride = HPRQP_GPU_FUSED_CHECK_THREADS / 2U;
         stride > 0U;
         stride >>= 1U) {
        if (tid < stride) {
            partial[0][tid] += partial[0][tid + stride];
            partial[1][tid] += partial[1][tid + stride];
            partial[2][tid] += partial[2][tid + stride];
            partial[3][tid] += partial[3][tid + stride];
            partial[4][tid] =
                fmax(partial[4][tid], partial[4][tid + stride]);
            partial[5][tid] =
                fmax(partial[5][tid], partial[5][tid + stride]);
            partial[6][tid] =
                fmax(partial[6][tid], partial[6][tid + stride]);
            partial[7][tid] =
                fmax(partial[7][tid], partial[7][tid + stride]);
            partial[8][tid] =
                fmax(partial[8][tid], partial[8][tid + stride]);
            partial[9][tid] =
                fmax(partial[9][tid], partial[9][tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0U) {
        for (unsigned int slot = 0U; slot < 10U; ++slot) {
            output[slot] = partial[slot][0];
        }
    }
}

__global__ static void gpu_finalize_rd_kernel(double *rd,
                                              double *scaled_qx_abs,
                                              double *scaled_atdy_abs,
                                              const double *qx,
                                              const double *c,
                                              const double *atdy,
                                              const double *z_bar,
                                              const double *col_norm,
                                              double c_scale,
                                              size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }

    double scale_fact = col_norm[i] * c_scale;
    double scaled_qx = qx[i] * scale_fact;
    double scaled_atdy = atdy[i] * scale_fact;

    rd[i] = (qx[i] + c[i] - atdy[i] - z_bar[i]) * scale_fact;
    scaled_qx_abs[i] = fabs(scaled_qx);
    scaled_atdy_abs[i] = fabs(scaled_atdy);
}

__global__ static void gpu_finalize_rd_qc_kernel(
    double *rd,
    double *scaled_qx_abs,
    double *scaled_aty_abs,
    double *scaled_bts_abs,
    const double *qx,
    const double *c,
    const double *aty,
    const double *bts,
    const double *z_bar,
    const double *col_norm,
    double c_scale,
    size_t n)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    if (i >= n) return;

    const double scale_fact = col_norm[i] * c_scale;
    rd[i] = (qx[i] + c[i] - aty[i] - bts[i] - z_bar[i]) *
            scale_fact;
    scaled_qx_abs[i] = fabs(qx[i] * scale_fact);
    scaled_aty_abs[i] = fabs(aty[i] * scale_fact);
    scaled_bts_abs[i] = fabs(bts[i] * scale_fact);
}

/*
 * The projected HPR iterates satisfy the variable box automatically, so the
 * box is not part of eta_primal after the first update.  At iteration zero,
 * however, x_bar is the unprojected initialization.  Match the CPU safeguard
 * and prevent that point from being certified before its first projection.
 */
__global__ static void gpu_initial_bound_violation_kernel(
    double *violation,
    const double *x_bar,
    const double *lower,
    const double *upper,
    const double *col_norm,
    double b_scale,
    size_t n)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    double value = 0.0;
    if (i >= n) return;

    if (lower[i] - x_bar[i] > 0.0) {
        value += lower[i] - x_bar[i];
    }
    if (x_bar[i] - upper[i] > 0.0) {
        value += x_bar[i] - upper[i];
    }
    violation[i] = fabs(value * (b_scale / col_norm[i]));
}

__global__ static void gpu_fill_ones_kernel(double *values, size_t length)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    if (i < length) values[i] = 1.0;
}

__global__ static void gpu_finalize_rp_kernel(double *rp,
                                              double *scaled_ax_abs,
                                              double *s,
                                              const double *ax,
                                              const double *al,
                                              const double *au,
                                              const double *row_norm,
                                              double b_scale,
                                              size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i >= m) {
        return;
    }

    double ax_i = ax[i];
    double projected = ax_i < al[i] ? al[i] : (ax_i > au[i] ? au[i] : ax_i);
    double scale_fact = row_norm[i] * b_scale;

    s[i] = projected;
    rp[i] = (projected - ax_i) * scale_fact;
    scaled_ax_abs[i] = fabs(ax_i * scale_fact);
}

HPRQPStatus gpu_compute_residuals_value(HPRQPGPUWorkspace *gpu_workspace,
                                        const HPRQPGPUProblem *gpu_problem,
                                        HPRQPResiduals *residuals,
                                        size_t iter)
{
    enum {
        SLOT_C_DOT_X = 0,
        SLOT_X_DOT_QX,
        SLOT_Z_DOT_X,
        SLOT_Y_DOT_S,
        SLOT_MAX_RD,
        SLOT_MAX_QX,
        SLOT_MAX_ATY,
        SLOT_MAX_RP,
        SLOT_MAX_AX,
        SLOT_H_DOT_QC_DUAL,
        SLOT_MAX_QC,
        SLOT_MAX_BX,
        SLOT_MAX_H,
        SLOT_MAX_BTS,
        SLOT_MAX_INITIAL_BOUND,
        SLOT_COUNT
    };
    cublasHandle_t blas_handle;
    HPRQPStatus status;
    HPRQPStatus restore_status;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;
    double c_dot_x;
    double x_dot_qx;
    double z_dot_x;
    double y_dot_s = 0.0;
    double max_scaled_qx = 0.0;
    double max_scaled_atdy = 0.0;
    double max_scaled_ax = 0.0;
    double h_dot_qc_dual = 0.0;
    double max_qc_bx = 0.0;
    double max_qc_h = 0.0;
    double max_scaled_bts = 0.0;
    double *device_values;
    int *device_indices;
    double host_values[SLOT_COUNT];
    int use_fused_parallel_check;

    memset(residuals, 0, sizeof(*residuals));
    memset(host_values, 0, sizeof(host_values));
    blas_handle = (cublasHandle_t)gpu_workspace->blas_handle;
    device_values = gpu_batch_reduce_values(gpu_workspace);
    device_indices = gpu_batch_reduce_indices(gpu_workspace);
    if (blas_handle == NULL || device_values == NULL ||
        device_indices == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    threads_n = gpu_workspace->threads_n;
    blocks_n = gpu_workspace->blocks_n;
    threads_m = gpu_workspace->threads_m;
    blocks_m = gpu_workspace->blocks_m;
    use_fused_parallel_check =
        gpu_workspace->method == HPRQP_METHOD_PARALLEL &&
        gpu_problem->p == 0U &&
        iter > 0U &&
        gpu_workspace->n + gpu_workspace->m <=
            (size_t)HPRQP_GPU_FUSED_CHECK_MAX_LENGTH;

    status = gpu_problem->Q.nnz > 0U
                 ? gpu_spmv_apply_q_x_bar(gpu_workspace, gpu_problem)
                 : map_cuda_status(cudaMemset(gpu_workspace->Qx.values, 0,
                                              gpu_workspace->n * sizeof(double)));
    if (status != HPRQP_STATUS_OK) return status;

    /* eta_primal and eta_QC use their own original-model denominators. */
    if (!gpu_workspace->empty_A) {
        status = gpu_spmv_apply_a_x_bar(gpu_workspace, gpu_problem);
        if (status != HPRQP_STATUS_OK) return status;
        if (!use_fused_parallel_check) {
            status = gpu_spmv_apply_at_y_bar_residual(gpu_workspace,
                                                      gpu_problem);
            if (status != HPRQP_STATUS_OK) return status;
        }
    } else if (gpu_problem->p > 0U) {
        status = map_cuda_status(cudaMemset(gpu_workspace->ATdy.values, 0,
                                            gpu_workspace->n * sizeof(double)));
        if (status != HPRQP_STATUS_OK) return status;
    }

    if (gpu_problem->p > 0U) {
        status = gpu_qcqp_apply_bt(gpu_workspace, gpu_problem,
                                   gpu_workspace->qc_dual_bar.values,
                                   gpu_workspace->BTs_bar.values);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_qcqp_compute_residual_vectors(gpu_workspace,
                                                   gpu_problem);
        if (status != HPRQP_STATUS_OK) return status;
    }

    if (use_fused_parallel_check) {
        gpu_parallel_fused_residual_stats_kernel
            <<<1U, HPRQP_GPU_FUSED_CHECK_THREADS>>>(
                device_values,
                gpu_workspace->Rd.values,
                gpu_workspace->Rp.values,
                gpu_workspace->reduce_tmp1.values,
                gpu_workspace->reduce_tmp2.values,
                gpu_workspace->s.values,
                gpu_workspace->x_bar.values,
                gpu_workspace->y_bar.values,
                gpu_workspace->z_bar.values,
                gpu_workspace->Qx.values,
                gpu_workspace->ATy_bar.values,
                gpu_workspace->Ax.values,
                gpu_problem->c.values,
                gpu_problem->l.values,
                gpu_problem->u.values,
                gpu_problem->AL.values,
                gpu_problem->AU.values,
                gpu_workspace->col_norm.values,
                gpu_workspace->row_norm.values,
                gpu_workspace->scale_c_scale,
                gpu_workspace->scale_b_scale,
                gpu_workspace->n,
                gpu_workspace->empty_A ? 0U : gpu_workspace->m,
                0);
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
        goto reductions_complete;
    }

    status = gpu_batch_reduce_begin(blas_handle);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_raw_dot_device(blas_handle,
                                gpu_workspace->n,
                                gpu_problem->c.values,
                                gpu_workspace->x_bar.values,
                                device_values + SLOT_C_DOT_X);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_raw_dot_device(blas_handle,
                                gpu_workspace->n,
                                gpu_workspace->x_bar.values,
                                gpu_workspace->Qx.values,
                                device_values + SLOT_X_DOT_QX);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_raw_dot_device(blas_handle,
                                gpu_workspace->n,
                                gpu_workspace->z_bar.values,
                                gpu_workspace->x_bar.values,
                                device_values + SLOT_Z_DOT_X);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;

    if (gpu_problem->p > 0U) {
        gpu_finalize_rd_qc_kernel<<<blocks_n, threads_n>>>(
            gpu_workspace->Rd.values, gpu_workspace->reduce_tmp1.values,
            gpu_workspace->reduce_tmp2.values,
            gpu_workspace->qc_scaled_bts_abs.values,
            gpu_workspace->Qx.values, gpu_problem->c.values,
            gpu_workspace->ATdy.values, gpu_workspace->BTs_bar.values,
            gpu_workspace->z_bar.values, gpu_workspace->col_norm.values,
            gpu_workspace->scale_c_scale, gpu_workspace->n);
    } else {
        gpu_finalize_rd_kernel<<<blocks_n, threads_n>>>(
            gpu_workspace->Rd.values, gpu_workspace->reduce_tmp1.values,
            gpu_workspace->reduce_tmp2.values, gpu_workspace->Qx.values,
            gpu_problem->c.values, gpu_workspace->ATdy.values,
            gpu_workspace->z_bar.values, gpu_workspace->col_norm.values,
            gpu_workspace->scale_c_scale, gpu_workspace->n);
    }
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_raw_iamax_device(blas_handle,
                                  gpu_workspace->n,
                                  gpu_workspace->Rd.values,
                                  device_indices + 0);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_raw_iamax_device(blas_handle,
                                  gpu_workspace->n,
                                  gpu_workspace->reduce_tmp1.values,
                                  device_indices + 1);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_raw_iamax_device(blas_handle,
                                  gpu_workspace->n,
                                  gpu_workspace->reduce_tmp2.values,
                                  device_indices + 2);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_batch_gather_abs(device_values,
                                  SLOT_MAX_RD,
                                  device_indices,
                                  0,
                                  gpu_workspace->Rd.values,
                                  gpu_workspace->n);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_batch_gather_abs(device_values,
                                  SLOT_MAX_QX,
                                  device_indices,
                                  1,
                                  gpu_workspace->reduce_tmp1.values,
                                  gpu_workspace->n);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    status = gpu_batch_gather_abs(device_values,
                                  SLOT_MAX_ATY,
                                  device_indices,
                                  2,
                                  gpu_workspace->reduce_tmp2.values,
                                  gpu_workspace->n);
    if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;

    if (iter == 0U) {
        gpu_initial_bound_violation_kernel<<<blocks_n, threads_n>>>(
            gpu_workspace->reduce_tmp1.values,
            gpu_workspace->x_bar.values,
            gpu_problem->l.values,
            gpu_problem->u.values,
            gpu_workspace->col_norm.values,
            gpu_workspace->scale_b_scale,
            gpu_workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_iamax_device(blas_handle,
                                      gpu_workspace->n,
                                      gpu_workspace->reduce_tmp1.values,
                                      device_indices + 5);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_batch_gather_abs(
            device_values, SLOT_MAX_INITIAL_BOUND, device_indices, 5,
            gpu_workspace->reduce_tmp1.values, gpu_workspace->n);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    }

    if (!gpu_workspace->empty_A) {
        gpu_finalize_rp_kernel<<<blocks_m, threads_m>>>(gpu_workspace->Rp.values,
                                                        gpu_workspace->reduce_tmp1.values,
                                                        gpu_workspace->s.values,
                                                        gpu_workspace->Ax.values,
                                                        gpu_problem->AL.values,
                                                        gpu_problem->AU.values,
                                                        gpu_workspace->row_norm.values,
                                                        gpu_workspace->scale_b_scale,
                                                        gpu_workspace->m);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_dot_device(blas_handle,
                                    gpu_workspace->m,
                                    gpu_workspace->y_bar.values,
                                    gpu_workspace->s.values,
                                    device_values + SLOT_Y_DOT_S);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_iamax_device(blas_handle,
                                      gpu_workspace->m,
                                      gpu_workspace->Rp.values,
                                      device_indices + 3);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_iamax_device(blas_handle,
                                      gpu_workspace->m,
                                      gpu_workspace->reduce_tmp1.values,
                                      device_indices + 4);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_batch_gather_abs(device_values,
                                      SLOT_MAX_RP,
                                      device_indices,
                                      3,
                                      gpu_workspace->Rp.values,
                                      gpu_workspace->m);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_batch_gather_abs(device_values,
                                      SLOT_MAX_AX,
                                      device_indices,
                                      4,
                                      gpu_workspace->reduce_tmp1.values,
                                      gpu_workspace->m);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    }

    if (gpu_problem->p > 0U) {
        status = gpu_raw_iamax_device(blas_handle,
                                      gpu_workspace->p,
                                      gpu_workspace->qc_primal_violation.values,
                                      device_indices + 0);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_iamax_device(blas_handle, gpu_workspace->p,
                                      gpu_workspace->qc_bx_norm.values,
                                      device_indices + 1);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_iamax_device(blas_handle, gpu_workspace->p,
                                      gpu_workspace->qc_h_norm.values,
                                      device_indices + 2);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_iamax_device(
            blas_handle, gpu_workspace->n,
            gpu_workspace->qc_scaled_bts_abs.values, device_indices + 3);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_batch_gather_abs(device_values,
                                      SLOT_MAX_QC,
                                      device_indices,
                                      0,
                                      gpu_workspace->qc_primal_violation.values,
                                      gpu_workspace->p);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_batch_gather_abs(device_values, SLOT_MAX_BX,
                                      device_indices, 1,
                                      gpu_workspace->qc_bx_norm.values,
                                      gpu_workspace->p);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_batch_gather_abs(device_values, SLOT_MAX_H,
                                      device_indices, 2,
                                      gpu_workspace->qc_h_norm.values,
                                      gpu_workspace->p);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_batch_gather_abs(
            device_values, SLOT_MAX_BTS, device_indices, 3,
            gpu_workspace->qc_scaled_bts_abs.values, gpu_workspace->n);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        gpu_fill_ones_kernel<<<gpu_workspace->blocks_p,
                               gpu_workspace->threads_p>>>(
            gpu_workspace->qc_h_norm.values, gpu_workspace->p);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
        status = gpu_raw_dot_device(
            blas_handle, gpu_workspace->p,
            gpu_workspace->qc_dual_objective.values,
            gpu_workspace->qc_h_norm.values,
            device_values + SLOT_H_DOT_QC_DUAL);
        if (status != HPRQP_STATUS_OK) goto restore_pointer_mode;
    }

    status = gpu_batch_reduce_copy_to_host(gpu_workspace,
                                           host_values,
                                           SLOT_COUNT);

restore_pointer_mode:
    restore_status = gpu_batch_reduce_end(blas_handle);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    if (restore_status != HPRQP_STATUS_OK) {
        return restore_status;
    }

reductions_complete:
    c_dot_x = host_values[SLOT_C_DOT_X];
    x_dot_qx = host_values[SLOT_X_DOT_QX];
    z_dot_x = host_values[SLOT_Z_DOT_X];
    y_dot_s = gpu_workspace->empty_A ? 0.0 : host_values[SLOT_Y_DOT_S];
    residuals->err_Rd_org_bar = host_values[SLOT_MAX_RD];
    max_scaled_qx = host_values[SLOT_MAX_QX];
    max_scaled_atdy = host_values[SLOT_MAX_ATY];
    residuals->err_Rp_org_bar =
        gpu_workspace->empty_A ? 0.0 : host_values[SLOT_MAX_RP];
    max_scaled_ax =
        gpu_workspace->empty_A ? 0.0 : host_values[SLOT_MAX_AX];
    if (gpu_problem->p > 0U) {
        h_dot_qc_dual = host_values[SLOT_H_DOT_QC_DUAL];
        residuals->err_QC_org_bar = host_values[SLOT_MAX_QC];
        max_qc_bx = host_values[SLOT_MAX_BX];
        max_qc_h = host_values[SLOT_MAX_H];
        max_scaled_bts = host_values[SLOT_MAX_BTS];
    }

    if (getenv("HPRQP_TRACE_RESIDUAL") != NULL) {
        fprintf(stderr,
                "HPRQP_RESIDUAL_RAW iter=%zu rp=%.17g ax=%.17g "
                "norm_b=%.17g qc=%.17g bx=%.17g h=%.17g "
                "rd=%.17g norm_c=%.17g aty=%.17g bts=%.17g\n",
                iter, residuals->err_Rp_org_bar, max_scaled_ax,
                gpu_workspace->scale_norm_b_org,
                gpu_problem->p > 0U ? residuals->err_QC_org_bar : 0.0,
                max_qc_bx, max_qc_h, residuals->err_Rd_org_bar,
                gpu_workspace->scale_norm_c_org, max_scaled_atdy,
                max_scaled_bts);
    }

    residuals->primal_obj_bar = gpu_workspace->scale_b_scale * gpu_workspace->scale_c_scale *
                                (c_dot_x + 0.5 * x_dot_qx) + gpu_problem->obj_constant;
    residuals->dual_obj_bar = gpu_workspace->scale_b_scale * gpu_workspace->scale_c_scale *
                              (-0.5 * x_dot_qx + z_dot_x + y_dot_s -
                               h_dot_qc_dual) + gpu_problem->obj_constant;
    residuals->rel_gap_bar = fabs(residuals->primal_obj_bar - residuals->dual_obj_bar) /
                             (1.0 + fmax(fabs(residuals->primal_obj_bar), fabs(residuals->dual_obj_bar)));
    residuals->err_Rd_org_bar /=
        1.0 + fmax(gpu_workspace->scale_norm_c_org,
                   fmax(max_scaled_bts,
                        fmax(max_scaled_atdy, max_scaled_qx)));

    if (!gpu_workspace->empty_A) {
        residuals->err_Rp_org_bar /=
            1.0 + fmax(gpu_workspace->scale_norm_b_org,
                       max_scaled_ax);
    }
    if (gpu_problem->p > 0U) {
        residuals->err_QC_org_bar /=
            1.0 + fmax(max_qc_h, max_qc_bx);
    }

    if (iter == 0U &&
        host_values[SLOT_MAX_INITIAL_BOUND] > residuals->err_Rp_org_bar) {
        residuals->err_Rp_org_bar = host_values[SLOT_MAX_INITIAL_BOUND];
    }

    residuals->KKTx_and_gap_org_bar = fmax(residuals->err_Rp_org_bar,
        fmax(residuals->err_QC_org_bar,
             fmax(residuals->err_Rd_org_bar, residuals->rel_gap_bar)));
    residuals->is_updated = 1;
    return HPRQP_STATUS_OK;
}
