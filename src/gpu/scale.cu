#include "cuda_common.h"

__global__ static void gpu_fill_vector_kernel(double *values, double scalar, size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    values[i] = scalar;
}

__global__ static void gpu_curtis_reid_log_update_kernel(
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *col_ind,
    const double *values,
    const double *opposite_log_scale,
    double *log_scale,
    size_t nrow)
{
    const size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t idx;
    size_t count;
    double sum = 0.0;

    if (row >= nrow) {
        return;
    }

    count = (size_t)(row_ptr[row + 1] - row_ptr[row]);
    for (idx = (size_t)row_ptr[row]; idx < (size_t)row_ptr[row + 1]; ++idx) {
        const double magnitude = fmax(fabs(values[idx]), 1e-300);
        sum += -log(magnitude) - opposite_log_scale[col_ind[idx]];
    }
    log_scale[row] = count > 0 ? sum / (double)count : 0.0;
}

__global__ static void gpu_curtis_reid_build_inverse_norm_kernel(
    double *log_scale_and_norm,
    size_t length)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double scale;

    if (i >= length) {
        return;
    }

    scale = exp(log_scale_and_norm[i]);
    scale = fmin(fmax(scale, 1e-30), 1e30);
    log_scale_and_norm[i] = 1.0 / scale;
}

/* Joint Curtis--Reid row update for one affine rotated-SOC block.  The
 * coordinate log multipliers are constrained to (a-b,a+b,a), so their
 * exponentials preserve 2uv >= ||t||^2 exactly.  The constant and offset are
 * included as entries in the homogeneous augmented operator [B h]. */
__global__ static void gpu_curtis_reid_qc_row_update_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *factor_row_ptr,
    const HPRQPGPUCsrIndex *factor_col_ind,
    const double *factor_values,
    const double *constant,
    const double *offset,
    const double *col_log_scale,
    double *a_log_scale,
    double *b_log_shape,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    double z0 = 0.0;
    double z1 = 0.0;
    double zt = 0.0;
    double w0 = 0.0;
    double w1 = 0.0;
    double wt = 0.0;

    if (cone >= p) {
        return;
    }

    if (constant[cone] != 0.0) {
        z0 = -log(fmax(fabs(constant[cone]), 1e-300));
        w0 = 1.0;
    }
    if (offset[cone] != 0.0) {
        z1 += -log(fmax(fabs(offset[cone]), 1e-300));
        w1 += 1.0;
    }
    for (size_t idx = (size_t)c_row_ptr[cone];
         idx < (size_t)c_row_ptr[cone + 1]; ++idx) {
        if (c_values[idx] != 0.0) {
            z1 += -log(fmax(fabs(c_values[idx]), 1e-300)) -
                  col_log_scale[c_col_ind[idx]];
            w1 += 1.0;
        }
    }
    for (size_t idx = (size_t)factor_row_ptr[cone];
         idx < (size_t)factor_row_ptr[cone + 1]; ++idx) {
        if (factor_values[idx] != 0.0) {
            zt += -log(fmax(fabs(factor_values[idx]), 1e-300)) -
                  col_log_scale[factor_col_ind[idx]];
            wt += 1.0;
        }
    }

    {
        const double n11 = w0 + w1 + wt;
        const double n12 = -w0 + w1;
        const double n22 = w0 + w1;
        const double rhs1 = z0 + z1 + zt;
        const double rhs2 = -z0 + z1;
        const double determinant = n11 * n22 - n12 * n12;
        if (determinant > 1e-15 * fmax(1.0, n11 * n22)) {
            a_log_scale[cone] =
                (rhs1 * n22 - rhs2 * n12) / determinant;
            b_log_shape[cone] =
                (n11 * rhs2 - n12 * rhs1) / determinant;
        } else if (w0 > 0.0) {
            /* Minimum-norm solution of a-b=z0 for a constant-only block. */
            a_log_scale[cone] = 0.5 * z0;
            b_log_shape[cone] = -0.5 * z0;
        } else {
            a_log_scale[cone] = 0.0;
            b_log_shape[cone] = 0.0;
        }
    }
}

__global__ static void gpu_curtis_reid_column_base_kernel(
    const HPRQPGPUCsrIndex *at_row_ptr,
    const HPRQPGPUCsrIndex *at_col_ind,
    const double *at_values,
    const double *a_row_log_scale,
    double *target_sum,
    double *target_count,
    size_t n)
{
    const size_t col = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    double sum = 0.0;
    double count = 0.0;

    if (col >= n) {
        return;
    }
    for (size_t idx = (size_t)at_row_ptr[col];
         idx < (size_t)at_row_ptr[col + 1]; ++idx) {
        if (at_values[idx] != 0.0) {
            sum += -log(fmax(fabs(at_values[idx]), 1e-300)) -
                   a_row_log_scale[at_col_ind[idx]];
            count += 1.0;
        }
    }
    target_sum[col] = sum;
    target_count[col] = count;
}

__global__ static void gpu_curtis_reid_accumulate_qc_columns_kernel(
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *col_ind,
    const double *values,
    const double *a_log_scale,
    const double *b_log_shape,
    int affine_coordinate,
    double *target_sum,
    double *target_count,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    double row_log;

    if (cone >= p) {
        return;
    }
    row_log = a_log_scale[cone];
    if (affine_coordinate) {
        row_log += b_log_shape[cone];
    }
    for (size_t idx = (size_t)row_ptr[cone];
         idx < (size_t)row_ptr[cone + 1]; ++idx) {
        if (values[idx] == 0.0) {
            continue;
        }
        atomicAdd(target_sum + col_ind[idx],
                  -log(fmax(fabs(values[idx]), 1e-300)) - row_log);
        atomicAdd(target_count + col_ind[idx], 1.0);
    }
}

__global__ static void gpu_curtis_reid_finalize_column_kernel(
    double *target_sum_and_log_scale,
    const double *target_count,
    size_t n)
{
    const size_t col = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    if (col >= n) {
        return;
    }
    target_sum_and_log_scale[col] = target_count[col] > 0.0
        ? target_sum_and_log_scale[col] / target_count[col]
        : 0.0;
}

__global__ static void gpu_apply_curtis_reid_qc_scaling_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    double *c_values,
    const HPRQPGPUCsrIndex *factor_row_ptr,
    double *factor_values,
    double *constant,
    double *offset,
    double *a_log_scale,
    double *shape_ratio,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    double a;
    double b;
    double max_log;
    double shrink;
    double s0;
    double s1;
    double st;

    if (cone >= p) {
        return;
    }
    a = a_log_scale[cone];
    b = shape_ratio[cone];
    max_log = fmax(fabs(a - b), fmax(fabs(a + b), fabs(a)));
    shrink = max_log > 69.07755278982137
                 ? 69.07755278982137 / max_log
                 : 1.0;
    a *= shrink;
    b *= shrink;
    s0 = exp(a - b);
    s1 = exp(a + b);
    st = exp(a);

    constant[cone] *= s0;
    offset[cone] *= s1;
    for (size_t idx = (size_t)c_row_ptr[cone];
         idx < (size_t)c_row_ptr[cone + 1]; ++idx) {
        c_values[idx] *= s1;
    }
    for (size_t idx = (size_t)factor_row_ptr[cone];
         idx < (size_t)factor_row_ptr[cone + 1]; ++idx) {
        factor_values[idx] *= st;
    }
    a_log_scale[cone] = a;
    /* s1/s0; used only to reconstruct original residuals and duals. */
    shape_ratio[cone] = exp(2.0 * b);
}

__global__ static void gpu_compute_row_max_abs_with_q_kernel(double *output,
                                                             const HPRQPGPUCsrIndex *at_row_ptr,
                                                             const double *at_values,
                                                             const HPRQPGPUCsrIndex *q_row_ptr,
                                                             const double *q_values,
                                                             size_t n)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double max_abs = 0.0;
    size_t idx;

    if (row >= n) {
        return;
    }

    for (idx = at_row_ptr[row]; idx < at_row_ptr[row + 1]; ++idx) {
        double value = fabs(at_values[idx]);
        if (value > max_abs) {
            max_abs = value;
        }
    }
    for (idx = q_row_ptr[row]; idx < q_row_ptr[row + 1]; ++idx) {
        double value = fabs(q_values[idx]);
        if (value > max_abs) {
            max_abs = value;
        }
    }

    output[row] = max_abs > 0.0 ? sqrt(max_abs) : 1.0;
}

__global__ static void gpu_compute_row_max_abs_kernel(double *output,
                                                      const HPRQPGPUCsrIndex *row_ptr,
                                                      const double *values,
                                                      size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double max_abs = 0.0;
    size_t idx;

    if (row >= nrow) {
        return;
    }

    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        double value = fabs(values[idx]);
        if (value > max_abs) {
            max_abs = value;
        }
    }

    output[row] = max_abs > 0.0 ? sqrt(max_abs) : 1.0;
}

__device__ static void gpu_atomic_max_nonnegative(double *address,
                                                  double value)
{
    unsigned long long int *bits =
        reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *bits;

    while (__longlong_as_double(old) < value) {
        const unsigned long long int assumed = old;
        old = atomicCAS(bits, assumed, __double_as_longlong(value));
        if (old == assumed) {
            break;
        }
    }
}

__global__ static void gpu_accumulate_qc_col_max_kernel(
    double *col_scale,
    const HPRQPGPUCsrIndex *col_ind,
    const double *values,
    size_t nnz)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    double operator_magnitude;
    double scale_candidate;

    if (idx >= nnz) {
        return;
    }
    operator_magnitude = fabs(values[idx]);
    scale_candidate = operator_magnitude > 0.0
                          ? sqrt(operator_magnitude)
                          : 0.0;
    gpu_atomic_max_nonnegative(col_scale + col_ind[idx], scale_candidate);
}

__global__ static void gpu_compute_row_sum_abs_with_q_kernel(double *output,
                                                             const HPRQPGPUCsrIndex *at_row_ptr,
                                                             const double *at_values,
                                                             const HPRQPGPUCsrIndex *q_row_ptr,
                                                             const double *q_values,
                                                             size_t n)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double sum_abs = 0.0;
    size_t idx;

    if (row >= n) {
        return;
    }

    for (idx = at_row_ptr[row]; idx < at_row_ptr[row + 1]; ++idx) {
        sum_abs += fabs(at_values[idx]);
    }
    for (idx = q_row_ptr[row]; idx < q_row_ptr[row + 1]; ++idx) {
        sum_abs += fabs(q_values[idx]);
    }

    output[row] = sum_abs > 0.0 ? sqrt(sum_abs) : 1.0;
}

__global__ static void gpu_compute_row_sum_abs_with_q_raw_kernel(
    double *output,
    const HPRQPGPUCsrIndex *at_row_ptr,
    const double *at_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const double *q_values,
    size_t n)
{
    const size_t row = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    double sum_abs = 0.0;

    if (row >= n) {
        return;
    }
    for (size_t idx = at_row_ptr[row]; idx < at_row_ptr[row + 1]; ++idx) {
        sum_abs += fabs(at_values[idx]);
    }
    for (size_t idx = q_row_ptr[row]; idx < q_row_ptr[row + 1]; ++idx) {
        sum_abs += fabs(q_values[idx]);
    }
    output[row] = sum_abs;
}

__global__ static void gpu_accumulate_qc_col_sum_kernel(
    double *col_sum,
    const HPRQPGPUCsrIndex *col_ind,
    const double *values,
    size_t nnz)
{
    const size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    double operator_magnitude;

    if (idx >= nnz) {
        return;
    }
    operator_magnitude = fabs(values[idx]);
    atomicAdd(col_sum + col_ind[idx], operator_magnitude);
}

__global__ static void gpu_finalize_col_sum_scale_kernel(double *col_scale,
                                                         size_t n)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    if (i >= n) {
        return;
    }
    col_scale[i] = col_scale[i] > 0.0 ? sqrt(col_scale[i]) : 1.0;
}

__global__ static void gpu_compute_row_sum_abs_kernel(double *output,
                                                      const HPRQPGPUCsrIndex *row_ptr,
                                                      const double *values,
                                                      size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    double sum_abs = 0.0;
    size_t idx;

    if (row >= nrow) {
        return;
    }

    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        sum_abs += fabs(values[idx]);
    }

    output[row] = sum_abs > 0.0 ? sqrt(sum_abs) : 1.0;
}

/* One Ruiz factor is shared by every coordinate of an affine cone block.
 * The first cone coordinate and the offset are part of the statistics even
 * though they do not belong to the linear operator B. */
__global__ static void gpu_compute_qc_block_max_scale_kernel(
    double *output,
    const HPRQPGPUCsrIndex *c_row_ptr,
    const double *c_values,
    const HPRQPGPUCsrIndex *factor_row_ptr,
    const double *factor_values,
    const double *constant,
    const double *offset,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    double max_abs;

    if (cone >= p) {
        return;
    }
    max_abs = fmax(fabs(constant[cone]), fabs(offset[cone]));
    for (size_t idx = (size_t)c_row_ptr[cone];
         idx < (size_t)c_row_ptr[cone + 1]; ++idx) {
        max_abs = fmax(max_abs, fabs(c_values[idx]));
    }
    for (size_t idx = (size_t)factor_row_ptr[cone];
         idx < (size_t)factor_row_ptr[cone + 1]; ++idx) {
        max_abs = fmax(max_abs, fabs(factor_values[idx]));
    }
    output[cone] = max_abs > 0.0 ? sqrt(max_abs) : 1.0;
}

/* Pock--Chambolle uses row 1-norms.  Under one scalar per cone, take the
 * largest coordinate-row statistic: |gamma| for the constant row,
 * |offset|+sum|c| for the affine row, and |R_j| for each tail row. */
__global__ static void gpu_compute_qc_block_sum_scale_kernel(
    double *output,
    const HPRQPGPUCsrIndex *c_row_ptr,
    const double *c_values,
    const HPRQPGPUCsrIndex *factor_row_ptr,
    const double *factor_values,
    const double *constant,
    const double *offset,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    double affine_sum;
    double max_row_sum;

    if (cone >= p) {
        return;
    }
    affine_sum = fabs(offset[cone]);
    for (size_t idx = (size_t)c_row_ptr[cone];
         idx < (size_t)c_row_ptr[cone + 1]; ++idx) {
        affine_sum += fabs(c_values[idx]);
    }
    max_row_sum = fmax(fabs(constant[cone]), affine_sum);
    for (size_t idx = (size_t)factor_row_ptr[cone];
         idx < (size_t)factor_row_ptr[cone + 1]; ++idx) {
        max_row_sum = fmax(max_row_sum, fabs(factor_values[idx]));
    }
    output[cone] = max_row_sum > 0.0 ? sqrt(max_row_sum) : 1.0;
}

__global__ static void gpu_scale_qc_blocks_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    double *c_values,
    const HPRQPGPUCsrIndex *factor_row_ptr,
    double *factor_values,
    double *constant,
    double *offset,
    const double *block_scale,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    double inverse;

    if (cone >= p) {
        return;
    }
    inverse = 1.0 / block_scale[cone];
    constant[cone] *= inverse;
    offset[cone] *= inverse;
    for (size_t idx = (size_t)c_row_ptr[cone];
         idx < (size_t)c_row_ptr[cone + 1]; ++idx) {
        c_values[idx] *= inverse;
    }
    for (size_t idx = (size_t)factor_row_ptr[cone];
         idx < (size_t)factor_row_ptr[cone + 1]; ++idx) {
        factor_values[idx] *= inverse;
    }
}

static void gpu_scale_qc_blocks(HPRQPGPUProblem *problem,
                                const double *block_scale,
                                unsigned int blocks,
                                unsigned int threads)
{
    if (problem->p == 0U) {
        return;
    }
    gpu_scale_qc_blocks_kernel<<<blocks, threads>>>(
        problem->qc_linear.row_ptr, problem->qc_linear.values,
        problem->qc_factor.row_ptr, problem->qc_factor.values,
        problem->qc_constant.values, problem->qc_offset.values,
        block_scale, problem->p);
}

__global__ static void gpu_scale_rows_csr_kernel(const HPRQPGPUCsrIndex *row_ptr,
                                                 double *values,
                                                 const double *row_scale,
                                                 size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t idx;
    double scale;

    if (row >= nrow) {
        return;
    }

    scale = 1.0 / row_scale[row];
    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        values[idx] *= scale;
    }
}

__global__ static void gpu_scale_cols_csr_kernel(const HPRQPGPUCsrIndex *row_ptr,
                                                 const HPRQPGPUCsrIndex *col_ind,
                                                 double *values,
                                                 const double *col_scale,
                                                 size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    size_t idx;

    if (row >= nrow) {
        return;
    }

    for (idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        values[idx] /= col_scale[col_ind[idx]];
    }
}

__global__ static void gpu_scale_qc_factor_cols_kernel(
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *col_ind,
    double *values,
    const double *col_scale,
    size_t nrow)
{
    size_t row = (size_t)blockIdx.x * (size_t)blockDim.x +
                 (size_t)threadIdx.x;
    if (row >= nrow) {
        return;
    }
    for (size_t idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx) {
        values[idx] /= col_scale[col_ind[idx]];
    }
}

static void gpu_scale_qc_columns(HPRQPGPUProblem *problem,
                                 const double *col_scale,
                                 unsigned int blocks,
                                 unsigned int threads)
{
    if (problem->p == 0) {
        return;
    }
    gpu_scale_cols_csr_kernel<<<blocks, threads>>>(
        problem->qc_linear.row_ptr,
        problem->qc_linear.col_ind,
        problem->qc_linear.values,
        col_scale,
        problem->p);
    gpu_scale_qc_factor_cols_kernel<<<blocks, threads>>>(
        problem->qc_factor.row_ptr,
        problem->qc_factor.col_ind,
        problem->qc_factor.values,
        col_scale,
        problem->p);
}

static void gpu_include_qc_in_col_max(HPRQPGPUProblem *problem,
                                      double *col_scale,
                                      unsigned int threads)
{
    if (problem->p == 0) {
        return;
    }
    if (problem->nnz_qc_linear > 0U) {
        const unsigned int blocks = (unsigned int)(
            (problem->nnz_qc_linear + threads - 1U) / threads);
        gpu_accumulate_qc_col_max_kernel<<<blocks, threads>>>(
            col_scale, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->nnz_qc_linear);
    }
    if (problem->nnz_qc_factor > 0U) {
        const unsigned int blocks = (unsigned int)(
            (problem->nnz_qc_factor + threads - 1U) / threads);
        gpu_accumulate_qc_col_max_kernel<<<blocks, threads>>>(
            col_scale, problem->qc_factor.col_ind,
            problem->qc_factor.values, problem->nnz_qc_factor);
    }
}

static void gpu_include_qc_in_col_sum(HPRQPGPUProblem *problem,
                                      double *col_sum,
                                      unsigned int blocks_n,
                                      unsigned int threads_n,
                                      size_t n)
{
    if (problem->p == 0) {
        return;
    }
    if (problem->nnz_qc_linear > 0U) {
        const unsigned int blocks = (unsigned int)(
            (problem->nnz_qc_linear + threads_n - 1U) / threads_n);
        gpu_accumulate_qc_col_sum_kernel<<<blocks, threads_n>>>(
            col_sum, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->nnz_qc_linear);
    }
    if (problem->nnz_qc_factor > 0U) {
        const unsigned int blocks = (unsigned int)(
            (problem->nnz_qc_factor + threads_n - 1U) / threads_n);
        gpu_accumulate_qc_col_sum_kernel<<<blocks, threads_n>>>(
            col_sum, problem->qc_factor.col_ind,
            problem->qc_factor.values, problem->nnz_qc_factor);
    }
    gpu_finalize_col_sum_scale_kernel<<<blocks_n, threads_n>>>(col_sum, n);
}

__global__ static void gpu_scale_vector_div_kernel(double *values,
                                                   const double *scale,
                                                   size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    values[i] /= scale[i];
}

__global__ static void gpu_scale_vector_mul_kernel(double *values,
                                                   const double *scale,
                                                   size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    if (i >= length) {
        return;
    }
    values[i] *= scale[i];
}

__global__ static void gpu_copy_finite_kernel(double *output,
                                              const double *input,
                                              size_t length)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    if (i >= length) {
        return;
    }
    output[i] = isfinite(input[i]) ? input[i] : 0.0;
}

__global__ static void gpu_qc_rhs_norm_component_kernel(
    double *output,
    const double *constant,
    const double *offset,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone >= p) {
        return;
    }
    output[cone] = hypot(constant[cone], offset[cone]);
}

__global__ static void gpu_scale_vector_scalar_kernel(double *values,
                                                      double scalar,
                                                      size_t length)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    if (i >= length) {
        return;
    }
    values[i] *= scalar;
}

__global__ static void gpu_scale_qc_rhs_scalar_kernel(double *constant,
                                                      double *offset,
                                                      double inverse_b_scale,
                                                      size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone >= p) {
        return;
    }
    constant[cone] *= inverse_b_scale;
    offset[cone] *= inverse_b_scale;
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

static HPRQPStatus gpu_compute_bound_max_local(cublasHandle_t handle,
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

    return gpu_raw_max_abs(handle, workspace->m, workspace->reduce_tmp1.values, result);
}

static HPRQPStatus gpu_compute_finite_norm2_local(
    cublasHandle_t handle,
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUVector *vector,
    double *result)
{
    HPRQPStatus status;

    if (workspace == NULL || vector == NULL || result == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (vector->length == 0U) {
        *result = 0.0;
        return HPRQP_STATUS_OK;
    }
    gpu_copy_finite_kernel<<<workspace->blocks_m, workspace->threads_m>>>(
        workspace->reduce_tmp1.values, vector->values, vector->length);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    return gpu_raw_norm2(handle, vector->length,
                         workspace->reduce_tmp1.values, result);
}

static HPRQPStatus gpu_compute_qc_rhs_norm2_local(
    cublasHandle_t handle,
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem,
    double *result)
{
    HPRQPStatus status;

    if (workspace == NULL || problem == NULL || result == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (problem->p == 0U) {
        *result = 0.0;
        return HPRQP_STATUS_OK;
    }
    gpu_qc_rhs_norm_component_kernel<<<workspace->blocks_p,
                                       workspace->threads_p>>>(
        workspace->qc_scale_step.values,
        problem->qc_constant.values,
        problem->qc_offset.values,
        problem->p);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    return gpu_raw_norm2(handle, problem->p,
                         workspace->qc_scale_step.values, result);
}

static double gpu_clip_bc_scale(double value)
{
    return fmin(fmax(value, 1e-4), 1e4);
}

static HPRQPStatus gpu_apply_bc_scaling(HPRQPGPUProblem *problem,
                                        HPRQPGPUWorkspace *workspace,
                                        cublasHandle_t handle,
                                        const HPRQPParams *params)
{
    HPRQPStatus status;
    double lower_norm = 0.0;
    double upper_norm = 0.0;
    double rhs_norm = 0.0;
    double objective_norm = 0.0;
    double b_scale;
    double c_scale;
    double inverse_b_scale;
    double inverse_c_scale;

    status = gpu_compute_finite_norm2_local(handle, workspace,
                                            &problem->AL, &lower_norm);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_compute_finite_norm2_local(handle, workspace,
                                            &problem->AU, &upper_norm);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_compute_qc_rhs_norm2_local(handle, workspace, problem,
                                            &rhs_norm);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_raw_norm2(handle, workspace->n, problem->c.values,
                           &objective_norm);
    if (status != HPRQP_STATUS_OK) return status;

    b_scale = gpu_clip_bc_scale(
        fmax(1.0, fmax(lower_norm, fmax(upper_norm, rhs_norm))));
    c_scale = gpu_clip_bc_scale(fmax(1.0, objective_norm));
    c_scale = fmin(fmax(c_scale, b_scale / 100.0), b_scale / 0.01);
    inverse_b_scale = 1.0 / b_scale;
    inverse_c_scale = 1.0 / c_scale;

    if (problem->Q.nnz > 0U) {
        const unsigned int threads = 256U;
        const unsigned int blocks = (unsigned int)(
            (problem->Q.nnz + threads - 1U) / threads);
        gpu_scale_vector_scalar_kernel<<<blocks, threads>>>(
            problem->Q.values, b_scale * inverse_c_scale, problem->Q.nnz);
    }
    if (workspace->m > 0U) {
        gpu_scale_vector_scalar_kernel<<<workspace->blocks_m,
                                         workspace->threads_m>>>(
            problem->AL.values, inverse_b_scale, workspace->m);
        gpu_scale_vector_scalar_kernel<<<workspace->blocks_m,
                                         workspace->threads_m>>>(
            problem->AU.values, inverse_b_scale, workspace->m);
    }
    if (workspace->n > 0U) {
        gpu_scale_vector_scalar_kernel<<<workspace->blocks_n,
                                         workspace->threads_n>>>(
            problem->c.values, inverse_c_scale, workspace->n);
        gpu_scale_vector_scalar_kernel<<<workspace->blocks_n,
                                         workspace->threads_n>>>(
            problem->l.values, inverse_b_scale, workspace->n);
        gpu_scale_vector_scalar_kernel<<<workspace->blocks_n,
                                         workspace->threads_n>>>(
            problem->u.values, inverse_b_scale, workspace->n);
    }
    if (problem->p > 0U) {
        gpu_scale_qc_rhs_scalar_kernel<<<workspace->blocks_p,
                                         workspace->threads_p>>>(
            problem->qc_constant.values,
            problem->qc_offset.values,
            inverse_b_scale,
            problem->p);
    }
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    workspace->scale_b_scale = b_scale;
    workspace->scale_c_scale = c_scale;
    if (params->verbose) {
        printf("HPR-SOCP b/c scaling: b_scale=%.15g c_scale=%.15g\n",
               b_scale, c_scale);
    }
    return HPRQP_STATUS_OK;
}

static HPRQPStatus gpu_download_scaling_to_host(HPRQPGPUWorkspace *workspace,
                                                HPRQPGPUProblem *gpu_problem,
                                                HPRQPScaling *host_scaling)
{
    cublasHandle_t handle;
    HPRQPStatus status;

    if (workspace == NULL || gpu_problem == NULL || host_scaling == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    handle = gpu_blas_handle(workspace);
    if (handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    scaling_free(host_scaling);
    host_scaling->b_scale = workspace->scale_b_scale;
    host_scaling->c_scale = workspace->scale_c_scale;
    host_scaling->norm_b_org = workspace->scale_norm_b_org;
    host_scaling->norm_c_org = workspace->scale_norm_c_org;

    if (workspace->n > 0) {
        status = gpu_raw_norm2(handle, workspace->n, gpu_problem->c.values, &host_scaling->norm_c);
        if (status != HPRQP_STATUS_OK) {
            scaling_free(host_scaling);
            return status;
        }
    }

    if (workspace->m > 0) {
        status = gpu_compute_bound_norm2_local(handle,
                                               workspace,
                                               &gpu_problem->AL,
                                               &gpu_problem->AU,
                                               &host_scaling->norm_b);
        if (status != HPRQP_STATUS_OK) {
            scaling_free(host_scaling);
            return status;
        }
    }

    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_problem_scale(HPRQPGPUProblem *gpu_problem,
                              HPRQPGPUWorkspace *workspace,
                              const HPRQPParams *params,
                              HPRQPScaling *host_scaling)
{
    cublasHandle_t handle;
    HPRQPStatus status;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;
    size_t iter;
    double initial_scaling_value;

    if (gpu_problem == NULL || workspace == NULL || params == NULL || host_scaling == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    handle = gpu_blas_handle(workspace);
    if (handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    threads_n = workspace->threads_n;
    blocks_n = workspace->blocks_n;
    threads_m = workspace->threads_m;
    blocks_m = workspace->blocks_m;

    workspace->scale_b_scale = 1.0;
    workspace->scale_c_scale = 1.0;
    status = gpu_compute_bound_max_local(handle,
                                         workspace,
                                         &gpu_problem->AL,
                                         &gpu_problem->AU,
                                         &workspace->scale_norm_b_org);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    status = gpu_raw_max_abs(handle,
                             workspace->n,
                             gpu_problem->c.values,
                             &workspace->scale_norm_c_org);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }

    initial_scaling_value = params->use_scaling && params->use_cr_scaling ? 0.0 : 1.0;
    if (gpu_launch_fill_vector(&workspace->col_norm, initial_scaling_value, threads_n) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    if (!workspace->empty_A &&
        gpu_launch_fill_vector(&workspace->row_norm, initial_scaling_value, threads_m) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    if (gpu_problem->p > 0U &&
        gpu_launch_fill_vector(&workspace->qc_cr_shape, 1.0,
                               workspace->threads_p) != HPRQP_STATUS_OK) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    if (params->use_scaling) {
        if (params->use_cr_scaling) {
            for (iter = 0; iter < params->cr_iterations; ++iter) {
                if (!workspace->empty_A && threads_m > 0U) {
                    gpu_curtis_reid_log_update_kernel<<<blocks_m, threads_m>>>(
                        gpu_problem->A.row_ptr,
                        gpu_problem->A.col_ind,
                        gpu_problem->A.values,
                        workspace->col_norm.values,
                        workspace->row_norm.values,
                        workspace->m);
                    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                        return HPRQP_STATUS_NOT_IMPLEMENTED;
                    }
                }
                if (gpu_problem->p > 0U) {
                    gpu_curtis_reid_qc_row_update_kernel<<<
                        workspace->blocks_p, workspace->threads_p>>>(
                        gpu_problem->qc_linear.row_ptr,
                        gpu_problem->qc_linear.col_ind,
                        gpu_problem->qc_linear.values,
                        gpu_problem->qc_factor.row_ptr,
                        gpu_problem->qc_factor.col_ind,
                        gpu_problem->qc_factor.values,
                        gpu_problem->qc_constant.values,
                        gpu_problem->qc_offset.values,
                        workspace->col_norm.values,
                        workspace->qc_scale_step.values,
                        workspace->qc_cr_shape.values,
                        gpu_problem->p);
                    if (cudaGetLastError() != cudaSuccess ||
                        cudaDeviceSynchronize() != cudaSuccess) {
                        return HPRQP_STATUS_NOT_IMPLEMENTED;
                    }
                }
                if (threads_n > 0U) {
                    gpu_curtis_reid_column_base_kernel<<<blocks_n, threads_n>>>(
                        gpu_problem->AT.row_ptr,
                        gpu_problem->AT.col_ind,
                        gpu_problem->AT.values,
                        workspace->row_norm.values,
                        workspace->col_norm.values,
                        workspace->diag_tmp.values,
                        workspace->n);
                    if (gpu_problem->p > 0U) {
                        gpu_curtis_reid_accumulate_qc_columns_kernel<<<
                            workspace->blocks_p, workspace->threads_p>>>(
                            gpu_problem->qc_linear.row_ptr,
                            gpu_problem->qc_linear.col_ind,
                            gpu_problem->qc_linear.values,
                            workspace->qc_scale_step.values,
                            workspace->qc_cr_shape.values,
                            1,
                            workspace->col_norm.values,
                            workspace->diag_tmp.values,
                            gpu_problem->p);
                        gpu_curtis_reid_accumulate_qc_columns_kernel<<<
                            workspace->blocks_p, workspace->threads_p>>>(
                            gpu_problem->qc_factor.row_ptr,
                            gpu_problem->qc_factor.col_ind,
                            gpu_problem->qc_factor.values,
                            workspace->qc_scale_step.values,
                            workspace->qc_cr_shape.values,
                            0,
                            workspace->col_norm.values,
                            workspace->diag_tmp.values,
                            gpu_problem->p);
                    }
                    gpu_curtis_reid_finalize_column_kernel<<<blocks_n, threads_n>>>(
                        workspace->col_norm.values,
                        workspace->diag_tmp.values,
                        workspace->n);
                    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                        return HPRQP_STATUS_NOT_IMPLEMENTED;
                    }
                }
            }

            if (!workspace->empty_A && threads_m > 0U) {
                gpu_curtis_reid_build_inverse_norm_kernel<<<blocks_m, threads_m>>>(
                    workspace->row_norm.values,
                    workspace->m);
            }
            if (threads_n > 0U) {
                gpu_curtis_reid_build_inverse_norm_kernel<<<blocks_n, threads_n>>>(
                    workspace->col_norm.values,
                    workspace->n);
            }
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                return HPRQP_STATUS_NOT_IMPLEMENTED;
            }

            if (gpu_problem->p > 0U) {
                gpu_apply_curtis_reid_qc_scaling_kernel<<<
                    workspace->blocks_p, workspace->threads_p>>>(
                    gpu_problem->qc_linear.row_ptr,
                    gpu_problem->qc_linear.values,
                    gpu_problem->qc_factor.row_ptr,
                    gpu_problem->qc_factor.values,
                    gpu_problem->qc_constant.values,
                    gpu_problem->qc_offset.values,
                    workspace->qc_scale_step.values,
                    workspace->qc_cr_shape.values,
                    gpu_problem->p);
                if (cudaGetLastError() != cudaSuccess ||
                    cudaDeviceSynchronize() != cudaSuccess) {
                    return HPRQP_STATUS_NOT_IMPLEMENTED;
                }
            }

            if (threads_n > 0U) {
                gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                                   gpu_problem->Q.values,
                                                                   workspace->col_norm.values,
                                                                   workspace->n);
                gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                                   gpu_problem->Q.col_ind,
                                                                   gpu_problem->Q.values,
                                                                   workspace->col_norm.values,
                                                                   workspace->n);
                gpu_scale_vector_div_kernel<<<blocks_n, threads_n>>>(gpu_problem->c.values,
                                                                     workspace->col_norm.values,
                                                                     workspace->n);
                gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->l.values,
                                                                     workspace->col_norm.values,
                                                                     workspace->n);
                gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->u.values,
                                                                     workspace->col_norm.values,
                                                                     workspace->n);
                gpu_scale_qc_columns(gpu_problem,
                                     workspace->col_norm.values,
                                     workspace->blocks_p,
                                     workspace->threads_p);
            }
            if (!workspace->empty_A && threads_m > 0U) {
                gpu_scale_rows_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                                   gpu_problem->A.values,
                                                                   workspace->row_norm.values,
                                                                   workspace->m);
                gpu_scale_cols_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                                   gpu_problem->A.col_ind,
                                                                   gpu_problem->A.values,
                                                                   workspace->col_norm.values,
                                                                   workspace->m);
                gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                                   gpu_problem->AT.values,
                                                                   workspace->col_norm.values,
                                                                   workspace->n);
                gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                                   gpu_problem->AT.col_ind,
                                                                   gpu_problem->AT.values,
                                                                   workspace->row_norm.values,
                                                                   workspace->n);
                gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AL.values,
                                                                     workspace->row_norm.values,
                                                                     workspace->m);
                gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AU.values,
                                                                     workspace->row_norm.values,
                                                                     workspace->m);
            }
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                return HPRQP_STATUS_NOT_IMPLEMENTED;
            }
        }

        for (iter = 0; iter < 10; ++iter) {
            if (threads_n > 0U) {
                gpu_compute_row_max_abs_with_q_kernel<<<blocks_n, threads_n>>>(workspace->reduce_tmp2.values,
                                                                                gpu_problem->AT.row_ptr,
                                                                                gpu_problem->AT.values,
                                                                                gpu_problem->Q.row_ptr,
                                                                                gpu_problem->Q.values,
                                                                                workspace->n);
                gpu_include_qc_in_col_max(gpu_problem,
                                          workspace->reduce_tmp2.values,
                                          threads_n);
            }
            if (!workspace->empty_A && threads_m > 0U) {
                gpu_compute_row_max_abs_kernel<<<blocks_m, threads_m>>>(workspace->reduce_tmp1.values,
                                                                        gpu_problem->A.row_ptr,
                                                                        gpu_problem->A.values,
                                                                        workspace->m);
            }
            if (gpu_problem->p > 0U) {
                gpu_compute_qc_block_max_scale_kernel<<<
                    workspace->blocks_p, workspace->threads_p>>>(
                    workspace->qc_scale_step.values,
                    gpu_problem->qc_linear.row_ptr,
                    gpu_problem->qc_linear.values,
                    gpu_problem->qc_factor.row_ptr,
                    gpu_problem->qc_factor.values,
                    gpu_problem->qc_constant.values,
                    gpu_problem->qc_offset.values,
                    gpu_problem->p);
            }
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                return HPRQP_STATUS_NOT_IMPLEMENTED;
            }

            if (threads_n > 0U) {
                gpu_pointwise_multiply_kernel<<<blocks_n, threads_n>>>(workspace->col_norm.values,
                                                                       workspace->col_norm.values,
                                                                       workspace->reduce_tmp2.values,
                                                                       workspace->n);
                gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                                   gpu_problem->Q.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->n);
                gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                                   gpu_problem->Q.col_ind,
                                                                   gpu_problem->Q.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->n);
                gpu_scale_vector_div_kernel<<<blocks_n, threads_n>>>(gpu_problem->c.values,
                                                                     workspace->reduce_tmp2.values,
                                                                     workspace->n);
                gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->l.values,
                                                                     workspace->reduce_tmp2.values,
                                                                     workspace->n);
                gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->u.values,
                                                                     workspace->reduce_tmp2.values,
                                                                     workspace->n);
                gpu_scale_qc_columns(gpu_problem,
                                     workspace->reduce_tmp2.values,
                                     workspace->blocks_p,
                                     workspace->threads_p);
            }

            if (!workspace->empty_A && threads_m > 0U) {
                gpu_pointwise_multiply_kernel<<<blocks_m, threads_m>>>(workspace->row_norm.values,
                                                                       workspace->row_norm.values,
                                                                       workspace->reduce_tmp1.values,
                                                                       workspace->m);
                gpu_scale_rows_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                                   gpu_problem->A.values,
                                                                   workspace->reduce_tmp1.values,
                                                                   workspace->m);
                gpu_scale_cols_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                                   gpu_problem->A.col_ind,
                                                                   gpu_problem->A.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->m);
                gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                                   gpu_problem->AT.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->n);
                gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                                   gpu_problem->AT.col_ind,
                                                                   gpu_problem->AT.values,
                                                                   workspace->reduce_tmp1.values,
                                                                   workspace->n);
                gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AL.values,
                                                                     workspace->reduce_tmp1.values,
                                                                     workspace->m);
                gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AU.values,
                                                                     workspace->reduce_tmp1.values,
                                                                     workspace->m);
            }
            gpu_scale_qc_blocks(gpu_problem,
                                workspace->qc_scale_step.values,
                                workspace->blocks_p,
                                workspace->threads_p);

            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                return HPRQP_STATUS_NOT_IMPLEMENTED;
            }
        }
        if (threads_n > 0U) {
            if (gpu_problem->p > 0U) {
                gpu_compute_row_sum_abs_with_q_raw_kernel<<<blocks_n, threads_n>>>(
                    workspace->reduce_tmp2.values,
                    gpu_problem->AT.row_ptr,
                    gpu_problem->AT.values,
                    gpu_problem->Q.row_ptr,
                    gpu_problem->Q.values,
                    workspace->n);
                gpu_include_qc_in_col_sum(gpu_problem,
                                          workspace->reduce_tmp2.values,
                                          blocks_n,
                                          threads_n,
                                          workspace->n);
            } else {
                gpu_compute_row_sum_abs_with_q_kernel<<<blocks_n, threads_n>>>(workspace->reduce_tmp2.values,
                                                                                gpu_problem->AT.row_ptr,
                                                                                gpu_problem->AT.values,
                                                                                gpu_problem->Q.row_ptr,
                                                                                gpu_problem->Q.values,
                                                                                workspace->n);
            }
        }
        if (!workspace->empty_A && threads_m > 0U) {
            gpu_compute_row_sum_abs_kernel<<<blocks_m, threads_m>>>(workspace->reduce_tmp1.values,
                                                                    gpu_problem->A.row_ptr,
                                                                    gpu_problem->A.values,
                                                                    workspace->m);
        }
        if (gpu_problem->p > 0U) {
            gpu_compute_qc_block_sum_scale_kernel<<<
                workspace->blocks_p, workspace->threads_p>>>(
                workspace->qc_scale_step.values,
                gpu_problem->qc_linear.row_ptr,
                gpu_problem->qc_linear.values,
                gpu_problem->qc_factor.row_ptr,
                gpu_problem->qc_factor.values,
                gpu_problem->qc_constant.values,
                gpu_problem->qc_offset.values,
                gpu_problem->p);
        }
        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }

        if (threads_n > 0U) {
            gpu_pointwise_multiply_kernel<<<blocks_n, threads_n>>>(workspace->col_norm.values,
                                                                   workspace->col_norm.values,
                                                                   workspace->reduce_tmp2.values,
                                                                   workspace->n);
            gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                               gpu_problem->Q.values,
                                                               workspace->reduce_tmp2.values,
                                                               workspace->n);
            gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->Q.row_ptr,
                                                               gpu_problem->Q.col_ind,
                                                               gpu_problem->Q.values,
                                                               workspace->reduce_tmp2.values,
                                                               workspace->n);
            gpu_scale_vector_div_kernel<<<blocks_n, threads_n>>>(gpu_problem->c.values,
                                                                 workspace->reduce_tmp2.values,
                                                                 workspace->n);
            gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->l.values,
                                                                 workspace->reduce_tmp2.values,
                                                                 workspace->n);
            gpu_scale_vector_mul_kernel<<<blocks_n, threads_n>>>(gpu_problem->u.values,
                                                                 workspace->reduce_tmp2.values,
                                                                 workspace->n);
            gpu_scale_qc_columns(gpu_problem,
                                 workspace->reduce_tmp2.values,
                                 workspace->blocks_p,
                                 workspace->threads_p);
        }

        if (!workspace->empty_A && threads_m > 0U) {
            gpu_pointwise_multiply_kernel<<<blocks_m, threads_m>>>(workspace->row_norm.values,
                                                                   workspace->row_norm.values,
                                                                   workspace->reduce_tmp1.values,
                                                                   workspace->m);
            gpu_scale_rows_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                               gpu_problem->A.values,
                                                               workspace->reduce_tmp1.values,
                                                               workspace->m);
            gpu_scale_cols_csr_kernel<<<blocks_m, threads_m>>>(gpu_problem->A.row_ptr,
                                                               gpu_problem->A.col_ind,
                                                               gpu_problem->A.values,
                                                               workspace->reduce_tmp2.values,
                                                               workspace->m);
            gpu_scale_rows_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                               gpu_problem->AT.values,
                                                               workspace->reduce_tmp2.values,
                                                               workspace->n);
            gpu_scale_cols_csr_kernel<<<blocks_n, threads_n>>>(gpu_problem->AT.row_ptr,
                                                               gpu_problem->AT.col_ind,
                                                               gpu_problem->AT.values,
                                                               workspace->reduce_tmp1.values,
                                                               workspace->n);
            gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AL.values,
                                                                 workspace->reduce_tmp1.values,
                                                                 workspace->m);
            gpu_scale_vector_div_kernel<<<blocks_m, threads_m>>>(gpu_problem->AU.values,
                                                                 workspace->reduce_tmp1.values,
                                                                 workspace->m);
        }
        gpu_scale_qc_blocks(gpu_problem,
                            workspace->qc_scale_step.values,
                            workspace->blocks_p,
                            workspace->threads_p);

        if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }

    }

    if (params->use_bc_scaling) {
        status = gpu_apply_bc_scaling(gpu_problem, workspace, handle, params);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
    }

    return gpu_download_scaling_to_host(workspace, gpu_problem, host_scaling);
}

/* Diagonal of the normalized primal HPR energy
 *
 *   A^*A/lambda_A + B^*D_B^{-1}B + Q/lambda_Q.
 *
 * Unlike coefficient equilibration, all three terms have the same
 * congruence under a diagonal change of primal variables. */
__global__ static void gpu_metric_diag_aq_kernel(
    double *metric_diag,
    const HPRQPGPUCsrIndex *at_row_ptr,
    const double *at_values,
    const double *diag_q,
    double inverse_lambda_a,
    double inverse_lambda_q,
    size_t n)
{
    const size_t col = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    double value = 0.0;
    if (col >= n) return;
    for (size_t idx = (size_t)at_row_ptr[col];
         idx < (size_t)at_row_ptr[col + 1U]; ++idx) {
        const double a = at_values[idx];
        value += a * a * inverse_lambda_a;
    }
    value += fmax(diag_q[col], 0.0) * inverse_lambda_q;
    metric_diag[col] = value;
}

__global__ static void gpu_metric_diag_b_kernel(
    double *metric_diag,
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *factor_row_ptr,
    const HPRQPGPUCsrIndex *factor_col_ind,
    const double *factor_values,
    const double *kappa,
    const double *theta,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone >= p) return;
    const double inv_kappa = kappa[cone] > 0.0 ? 1.0 / kappa[cone] : 0.0;
    const double inv_theta = theta[cone] > 0.0 ? 1.0 / theta[cone] : 0.0;
    for (size_t idx = (size_t)c_row_ptr[cone];
         idx < (size_t)c_row_ptr[cone + 1U]; ++idx) {
        const double c = c_values[idx];
        atomicAdd(metric_diag + c_col_ind[idx],
                  c * c * inv_theta * inv_kappa);
    }
    for (size_t idx = (size_t)factor_row_ptr[cone];
         idx < (size_t)factor_row_ptr[cone + 1U]; ++idx) {
        const double r = factor_values[idx];
        atomicAdd(metric_diag + factor_col_ind[idx], r * r * inv_kappa);
    }
}

__global__ static void gpu_finalize_metric_scale_kernel(
    double *scale,
    double *cumulative_scale,
    double damping,
    size_t n)
{
    const size_t col = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    if (col >= n) return;
    const double energy = scale[col];
    double step = energy > 1e-16 ? pow(energy, 0.5 * damping) : 1.0;
    step = fmin(fmax(step, 0.25), 4.0);
    scale[col] = step;
    cumulative_scale[col] *= step;
}

HPRQPStatus gpu_problem_hpr_metric_scale(HPRQPGPUProblem *problem,
                                         HPRQPGPUWorkspace *workspace,
                                         HPRQPScaling *host_scaling,
                                         double damping)
{
    HPRQPStatus status;
    double inverse_lambda_a;
    double inverse_lambda_q;

    if (problem == NULL || workspace == NULL || host_scaling == NULL ||
        workspace->n == 0U || !(damping > 0.0) || damping > 1.0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    inverse_lambda_a = workspace->lambda_max_A > 0.0
                           ? 1.0 / workspace->lambda_max_A
                           : 0.0;
    inverse_lambda_q = workspace->lambda_max_Q > 0.0
                           ? 1.0 / workspace->lambda_max_Q
                           : 0.0;

    gpu_metric_diag_aq_kernel<<<workspace->blocks_n, workspace->threads_n>>>(
        workspace->reduce_tmp2.values,
        problem->AT.row_ptr,
        problem->AT.values,
        workspace->diag_Q.values,
        inverse_lambda_a,
        inverse_lambda_q,
        workspace->n);
    if (problem->p > 0U) {
        gpu_metric_diag_b_kernel<<<workspace->blocks_p,
                                   workspace->threads_p>>>(
            workspace->reduce_tmp2.values,
            problem->qc_linear.row_ptr,
            problem->qc_linear.col_ind,
            problem->qc_linear.values,
            problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind,
            problem->qc_factor.values,
            workspace->qc_kappa.values,
            workspace->qc_theta.values,
            workspace->p);
    }
    gpu_finalize_metric_scale_kernel<<<workspace->blocks_n,
                                       workspace->threads_n>>>(
        workspace->reduce_tmp2.values,
        workspace->col_norm.values,
        damping,
        workspace->n);
    if (cudaGetLastError() != cudaSuccess) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }

    gpu_scale_rows_csr_kernel<<<workspace->blocks_n, workspace->threads_n>>>(
        problem->Q.row_ptr,
        problem->Q.values,
        workspace->reduce_tmp2.values,
        workspace->n);
    gpu_scale_cols_csr_kernel<<<workspace->blocks_n, workspace->threads_n>>>(
        problem->Q.row_ptr,
        problem->Q.col_ind,
        problem->Q.values,
        workspace->reduce_tmp2.values,
        workspace->n);
    gpu_scale_vector_div_kernel<<<workspace->blocks_n, workspace->threads_n>>>(
        problem->c.values,
        workspace->reduce_tmp2.values,
        workspace->n);
    gpu_scale_vector_mul_kernel<<<workspace->blocks_n, workspace->threads_n>>>(
        problem->l.values,
        workspace->reduce_tmp2.values,
        workspace->n);
    gpu_scale_vector_mul_kernel<<<workspace->blocks_n, workspace->threads_n>>>(
        problem->u.values,
        workspace->reduce_tmp2.values,
        workspace->n);
    gpu_scale_qc_columns(problem,
                         workspace->reduce_tmp2.values,
                         workspace->blocks_p,
                         workspace->threads_p);
    if (!workspace->empty_A) {
        gpu_scale_cols_csr_kernel<<<workspace->blocks_m,
                                    workspace->threads_m>>>(
            problem->A.row_ptr,
            problem->A.col_ind,
            problem->A.values,
            workspace->reduce_tmp2.values,
            workspace->m);
        gpu_scale_rows_csr_kernel<<<workspace->blocks_n,
                                    workspace->threads_n>>>(
            problem->AT.row_ptr,
            problem->AT.values,
            workspace->reduce_tmp2.values,
            workspace->n);
    }
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    fprintf(stderr,
            "HPRQP_HPR_METRIC_SCALING damping=%.17g "
            "lambda_A_raw=%.17g lambda_Q_raw=%.17g\n",
            damping,
            workspace->lambda_max_A,
            workspace->lambda_max_Q);
    status = gpu_download_scaling_to_host(workspace, problem, host_scaling);
    return status;
}
