#include "gpu/qcqp.h"

#include "cuda_common.h"
#include "gpu/spmv.h"
#include "shared/parallel.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

__device__ static double qc_q_value_for_col(
    const HPRQPGPUCsrIndex *col_ind,
    const double *values,
    HPRQPGPUCsrIndex begin,
    HPRQPGPUCsrIndex end,
    HPRQPGPUCsrIndex col)
{
    HPRQPGPUCsrIndex left = begin;
    HPRQPGPUCsrIndex right = end;
    while (left < right) {
        HPRQPGPUCsrIndex mid = left + (right - left) / 2;
        HPRQPGPUCsrIndex candidate = col_ind[mid];
        if (candidate < col) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    if (left < end && col_ind[left] == col) {
        return values[left] * values[left];
    }
    return 0.0;
}

__global__ static void qc_fill_kernel(double *values,
                                      double scalar,
                                      size_t length)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    if (i < length) {
        values[i] = scalar;
    }
}

/* Build a graph-Laplacian majorizer for the normalized B-Q coupling
 *
 *   C = D_B^{-1/2} B (Q / lambda_Q)^{1/2}.
 *
 * The affine row of each cone receives its absolute row sum, every factor
 * row is a singleton for the diagonal-QC representation, and each primal
 * coordinate receives its absolute column sum.  Aggregating the row weights
 * by their maximum inside a cone preserves the one-metric-per-cone resolvent.
 * The resulting block matrix diag(r, c) - [0 C; C^T 0] is PSD because it is
 * a sum of signed two-node graph Laplacians. */
__global__ static void qc_diag_bq_weights_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *factor_row_ptr,
    const HPRQPGPUCsrIndex *factor_col_ind,
    const double *factor_values,
    const double *diag_q,
    const double *kappa,
    double inv_sqrt_lambda_q,
    double *cone_row_weight,
    double *column_weight,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    double affine_row_sum = 0.0;
    double factor_row_max = 0.0;
    double inv_sqrt_kappa;
    if (cone >= p) return;
    inv_sqrt_kappa = 1.0 / sqrt(fmax(kappa[cone], 1e-300));
    for (HPRQPGPUCsrIndex index = c_row_ptr[cone];
         index < c_row_ptr[cone + 1U]; ++index) {
        const HPRQPGPUCsrIndex col = c_col_ind[index];
        const double value = fabs(c_values[index]) *
            sqrt(fmax(diag_q[col], 0.0)) * inv_sqrt_lambda_q *
            inv_sqrt_kappa;
        affine_row_sum += value;
        atomicAdd(column_weight + col, value);
    }
    for (HPRQPGPUCsrIndex index = factor_row_ptr[cone];
         index < factor_row_ptr[cone + 1U]; ++index) {
        const HPRQPGPUCsrIndex col = factor_col_ind[index];
        const double value = fabs(factor_values[index]) *
            sqrt(fmax(diag_q[col], 0.0)) * inv_sqrt_lambda_q *
            inv_sqrt_kappa;
        factor_row_max = fmax(factor_row_max, value);
        atomicAdd(column_weight + col, value);
    }
    cone_row_weight[cone] = fmax(affine_row_sum, factor_row_max);
}

__global__ static void qc_diag_bq_refresh_cone_kernel(
    const double *row_weight,
    double *cone_metric,
    double base_metric,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone < p) cone_metric[cone] = base_metric + row_weight[cone];
}

__global__ static void qc_diag_bq_refresh_w_kernel(
    const double *column_weight,
    double *w_gamma,
    double *w_inv,
    double sigma_lambda_q,
    double base_metric,
    size_t n)
{
    const size_t col = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    if (col < n) {
        const double gamma = sigma_lambda_q *
                             (base_metric + column_weight[col]);
        w_gamma[col] = gamma;
        w_inv[col] = 1.0 / (1.0 + gamma);
    }
}

__global__ static void qc_diag_bq_cone_energy_kernel(
    const double *cone_metric,
    const double *base_energy,
    double *weighted_energy,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone < p) weighted_energy[cone] =
        cone_metric[cone] * base_energy[cone];
}

__global__ static void qc_diag_bq_w_energy_kernel(
    const double *w_gamma,
    const double *dw,
    const double *dqw,
    double inverse_sigma,
    double *weighted_energy,
    size_t n)
{
    const size_t col = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    if (col < n) {
        const double value = w_gamma[col] * inverse_sigma *
                             dw[col] * dqw[col];
        weighted_energy[col] = fmax(value, 0.0);
    }
}

__global__ static void qc_scale_copy_kernel(double *output,
                                            const double *input,
                                            double scale,
                                            size_t length)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    if (i < length) {
        output[i] = scale * input[i];
    }
}

__global__ static void qc_probe_kernel(double *values, size_t length)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    if (i < length) {
        values[i] = 1.0 + ((double)i + 1.0) / ((double)length + 1.0);
    }
}

__global__ static void qc_eigen_residual_kernel(double *residual,
                                                const double *image,
                                                const double *vector,
                                                double eigenvalue,
                                                size_t length)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    if (i < length) {
        residual[i] = image[i] - eigenvalue * vector[i];
    }
}

__global__ static void qc_incidence_sum_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const double *theta,
    double *incidence_sum,
    size_t p)
{
    size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                  (size_t)threadIdx.x;
    if (cone >= p) {
        return;
    }

    HPRQPGPUCsrIndex ci = c_row_ptr[cone];
    HPRQPGPUCsrIndex ce = c_row_ptr[cone + 1];
    HPRQPGPUCsrIndex qi = q_row_ptr[cone];
    HPRQPGPUCsrIndex qe = q_row_ptr[cone + 1];
    const double inv_theta = 1.0 / theta[cone];

    while (ci < ce || qi < qe) {
        HPRQPGPUCsrIndex c_col = ci < ce ? c_col_ind[ci] : INT32_MAX;
        HPRQPGPUCsrIndex q_col = qi < qe ? q_col_ind[qi] : INT32_MAX;
        HPRQPGPUCsrIndex col = c_col < q_col ? c_col : q_col;
        double c = 0.0;
        double q = 0.0;
        if (ci < ce && c_col == col) {
            c = c_values[ci++];
        }
        if (qi < qe && q_col == col) {
            const double factor = q_values[qi++];
            q = factor * factor;
        }
        atomicAdd(incidence_sum + col, sqrt(c * c * inv_theta + q));
    }
}

__device__ static double qc_secular_value(
    double lambda,
    double inv_theta,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    HPRQPGPUCsrIndex cb,
    HPRQPGPUCsrIndex ce,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    HPRQPGPUCsrIndex qb,
    HPRQPGPUCsrIndex qe)
{
    double value = 1.0;
    for (HPRQPGPUCsrIndex ci = cb; ci < ce; ++ci) {
        const double q = qc_q_value_for_col(q_col_ind, q_values, qb, qe,
                                            c_col_ind[ci]);
        value -= c_values[ci] * c_values[ci] * inv_theta /
                 (lambda - q);
    }
    return value;
}

__global__ static void qc_kappa_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const double *theta,
    const double *incidence_sum,
    double *kappa,
    size_t p)
{
    size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                  (size_t)threadIdx.x;
    if (cone >= p) {
        return;
    }

    const HPRQPGPUCsrIndex cb = c_row_ptr[cone];
    const HPRQPGPUCsrIndex ce = c_row_ptr[cone + 1];
    const HPRQPGPUCsrIndex qb = q_row_ptr[cone];
    const HPRQPGPUCsrIndex qe = q_row_ptr[cone + 1];
    const double inv_theta = 1.0 / theta[cone];
    double max_q = 0.0;
    double c_norm_sq = 0.0;

    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        max_q = fmax(max_q, q_values[qi] * q_values[qi]);
    }
    for (HPRQPGPUCsrIndex ci = cb; ci < ce; ++ci) {
        c_norm_sq += c_values[ci] * c_values[ci] * inv_theta;
    }

    double beta = max_q;
    if (c_norm_sq > 0.0) {
        double lower = max_q;
        double upper = max_q + c_norm_sq;
        double probe = lower + fmax(1e-14, 1e-14 * fmax(1.0, fabs(lower)));
        if (probe < upper &&
            qc_secular_value(probe, inv_theta, c_col_ind, c_values, cb, ce,
                              q_col_ind, q_values, qb, qe) < 0.0) {
            lower = probe;
            for (int iter = 0; iter < 52; ++iter) {
                const double mid = 0.5 * (lower + upper);
                const double value = qc_secular_value(
                    mid, inv_theta, c_col_ind, c_values, cb, ce,
                    q_col_ind, q_values, qb, qe);
                if (value < 0.0) {
                    lower = mid;
                } else {
                    upper = mid;
                }
            }
            beta = upper;
        }
    }

    HPRQPGPUCsrIndex ci = cb;
    HPRQPGPUCsrIndex qi = qb;
    double overlap = 0.0;
    while (ci < ce || qi < qe) {
        HPRQPGPUCsrIndex c_col = ci < ce ? c_col_ind[ci] : INT32_MAX;
        HPRQPGPUCsrIndex q_col = qi < qe ? q_col_ind[qi] : INT32_MAX;
        HPRQPGPUCsrIndex col = c_col < q_col ? c_col : q_col;
        double c = 0.0;
        double q = 0.0;
        if (ci < ce && c_col == col) c = c_values[ci++];
        if (qi < qe && q_col == col) {
            const double factor = q_values[qi++];
            q = factor * factor;
        }
        const double s = sqrt(c * c * inv_theta + q);
        overlap += s * fmax(0.0, incidence_sum[col] - s);
    }

    kappa[cone] = fmax((beta + overlap) * (1.0 + 1e-12), 1e-16);
}

/* Apply B^* D_B^{-1} B without materializing B or D_B. */
__global__ static void qc_apply_normalized_gram_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const double *kappa,
    const double *theta,
    const double *input,
    double *output,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone >= p) return;

    const double inv_kappa = 1.0 / kappa[cone];
    double linear = 0.0;
    for (HPRQPGPUCsrIndex ci = c_row_ptr[cone];
         ci < c_row_ptr[cone + 1]; ++ci) {
        linear += c_values[ci] * input[c_col_ind[ci]];
    }
    linear *= inv_kappa / theta[cone];
    for (HPRQPGPUCsrIndex ci = c_row_ptr[cone];
         ci < c_row_ptr[cone + 1]; ++ci) {
        atomicAdd(output + c_col_ind[ci], c_values[ci] * linear);
    }
    for (HPRQPGPUCsrIndex qi = q_row_ptr[cone];
         qi < q_row_ptr[cone + 1]; ++qi) {
        const double factor = q_values[qi];
        atomicAdd(output + q_col_ind[qi],
                  factor * factor * input[q_col_ind[qi]] * inv_kappa);
    }
}

/* Apply B^* D_B^{-1/2} to a vector in the whitened cone coordinates. */
__global__ static void qc_apply_whitened_bt_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const double *kappa,
    const double *theta,
    const double *input,
    double *output,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone >= p) return;

    const HPRQPGPUCsrIndex qb = q_row_ptr[cone];
    const HPRQPGPUCsrIndex qe = q_row_ptr[cone + 1];
    const size_t start = 2U * cone + (size_t)qb;
    const double inv_sqrt_kappa = 1.0 / sqrt(kappa[cone]);
    const double second = input[start + 1U] * inv_sqrt_kappa /
                          sqrt(theta[cone]);

    for (HPRQPGPUCsrIndex ci = c_row_ptr[cone];
         ci < c_row_ptr[cone + 1]; ++ci) {
        atomicAdd(output + c_col_ind[ci], -c_values[ci] * second);
    }
    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        const size_t tail = start + 2U + (size_t)(qi - qb);
        atomicAdd(output + q_col_ind[qi],
                  q_values[qi] * input[tail] * inv_sqrt_kappa);
    }
}

/* Apply D_B^{-1/2} B to a primal-space vector. */
__global__ static void qc_apply_whitened_b_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const double *kappa,
    const double *theta,
    const double *input,
    double *output,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone >= p) return;

    const HPRQPGPUCsrIndex qb = q_row_ptr[cone];
    const HPRQPGPUCsrIndex qe = q_row_ptr[cone + 1];
    const size_t start = 2U * cone + (size_t)qb;
    const double inv_sqrt_kappa = 1.0 / sqrt(kappa[cone]);
    double linear = 0.0;

    for (HPRQPGPUCsrIndex ci = c_row_ptr[cone];
         ci < c_row_ptr[cone + 1]; ++ci) {
        linear += c_values[ci] * input[c_col_ind[ci]];
    }
    output[start] = 0.0;
    output[start + 1U] = -linear * inv_sqrt_kappa /
                         sqrt(theta[cone]);
    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        const size_t tail = start + 2U + (size_t)(qi - qb);
        output[tail] = q_values[qi] * input[q_col_ind[qi]] *
                       inv_sqrt_kappa;
    }
}

__global__ static void qc_bt_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const double *dual,
    double *output,
    size_t p)
{
    size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                  (size_t)threadIdx.x;
    if (cone >= p) {
        return;
    }

    const HPRQPGPUCsrIndex qb = q_row_ptr[cone];
    const HPRQPGPUCsrIndex qe = q_row_ptr[cone + 1];
    const size_t start = 2 * cone + (size_t)qb;
    const double second = dual[start + 1];
    for (HPRQPGPUCsrIndex ci = c_row_ptr[cone];
         ci < c_row_ptr[cone + 1]; ++ci) {
        atomicAdd(output + c_col_ind[ci], -c_values[ci] * second);
    }
    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        const size_t tail = start + 2 + (size_t)(qi - qb);
        atomicAdd(output + q_col_ind[qi], q_values[qi] * dual[tail]);
    }
}

__global__ static void qc_bt_gather_kernel(
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *dual_ind,
    const double *values,
    const double *dual,
    double *output,
    size_t n)
{
    const size_t row = (size_t)blockIdx.x * (size_t)blockDim.x +
                       (size_t)threadIdx.x;
    double sum = 0.0;
    if (row >= n) return;
    for (HPRQPGPUCsrIndex k = row_ptr[row]; k < row_ptr[row + 1U]; ++k) {
        sum += values[k] * dual[dual_ind[k]];
    }
    output[row] = sum;
}

__device__ static void project_rotated_soc(double *a,
                                           double *b,
                                           double tail_norm,
                                           double *tail_scale)
{
    const double inv_sqrt_two = 0.707106781186547524400844362104849039;
    double t = (*a + *b) * inv_sqrt_two;
    double u0 = (*a - *b) * inv_sqrt_two;
    double unorm = sqrt(u0 * u0 + tail_norm * tail_norm);

    if (unorm <= t) {
        *tail_scale = 1.0;
        return;
    }
    if (unorm <= -t) {
        *a = 0.0;
        *b = 0.0;
        *tail_scale = 0.0;
        return;
    }

    const double projected_t = 0.5 * (t + unorm);
    const double scale = unorm > 0.0 ? projected_t / unorm : 0.0;
    u0 *= scale;
    *a = (projected_t + u0) * inv_sqrt_two;
    *b = (projected_t - u0) * inv_sqrt_two;
    *tail_scale = scale;
}

__global__ static void qc_residual_vectors_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const double *constant,
    const double *offset,
    const double *cr_shape,
    const double *x_bar,
    const double *dual_bar,
    double *primal_violation,
    double *bx_norm,
    double *h_norm,
    double *dual_objective,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone >= p) return;

    const HPRQPGPUCsrIndex qb = q_row_ptr[cone];
    const HPRQPGPUCsrIndex qe = q_row_ptr[cone + 1];
    const size_t start = 2U * cone + (size_t)qb;
    double linear = 0.0;
    double tail_norm_sq = 0.0;
    double tail_max = 0.0;
    const double common = constant[cone];
    const double inverse_common = 1.0 / common;
    const double shape = cr_shape[cone];
    const double inverse_affine_scale = inverse_common / shape;
    const double inverse_tail_scale = inverse_common / sqrt(shape);

    for (HPRQPGPUCsrIndex ci = c_row_ptr[cone];
         ci < c_row_ptr[cone + 1]; ++ci) {
        linear += c_values[ci] * x_bar[c_col_ind[ci]];
    }
    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        const double tail = q_values[qi] * x_bar[q_col_ind[qi]];
        tail_norm_sq += tail * tail;
        tail_max = fmax(tail_max, fabs(tail));
    }

    double projected0 = 1.0;
    double projected1 = (-linear - offset[cone]) * inverse_affine_scale;
    const double original1 = projected1;
    double tail_scale;
    project_rotated_soc(&projected0, &projected1,
                        sqrt(tail_norm_sq) * inverse_tail_scale,
                        &tail_scale);

    const double residual0 = 1.0 - projected0;
    const double residual1 = original1 - projected1;
    double violation_rsoc = fmax(fabs(residual0), fabs(residual1));
    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        const double tail = q_values[qi] * x_bar[q_col_ind[qi]] *
                            inverse_tail_scale;
        const double tail_residual = fabs((1.0 - tail_scale) * tail);
        violation_rsoc = fmax(violation_rsoc, tail_residual);
    }

    primal_violation[cone] = violation_rsoc;
    bx_norm[cone] = fmax(fabs(linear) * inverse_affine_scale,
                         tail_max * inverse_tail_scale);
    h_norm[cone] = fmax(1.0, fabs(offset[cone]) * inverse_affine_scale);
    dual_objective[cone] =
        common * dual_bar[start] -
        offset[cone] * dual_bar[start + 1U];
}

__global__ static void qc_metric_energy_kernel(
    const HPRQPGPUCsrIndex *q_row_ptr,
    const double *kappa,
    const double *theta,
    const double *dual_delta,
    double *metric_energy,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone >= p) return;

    const HPRQPGPUCsrIndex qb = q_row_ptr[cone];
    const HPRQPGPUCsrIndex qe = q_row_ptr[cone + 1];
    const size_t start = 2U * cone + (size_t)qb;
    const double shape = theta[cone];
    double energy = dual_delta[start] * dual_delta[start] / shape +
                    shape * dual_delta[start + 1U] *
                        dual_delta[start + 1U];
    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        const double value =
            dual_delta[start + 2U + (size_t)(qi - qb)];
        energy += value * value;
    }
    metric_energy[cone] = kappa[cone] * energy;
}

__global__ static void qc_displacement_theta_kernel(
    const HPRQPGPUCsrIndex *q_row_ptr,
    const double *dual_delta,
    double *theta_raw,
    size_t p)
{
    const size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    if (cone < p) {
        const size_t start = 2U * cone + (size_t)q_row_ptr[cone];
        const double first = fabs(dual_delta[start]);
        const double second = fabs(dual_delta[start + 1U]);
        theta_raw[cone] = first > 1e-30 && second > 1e-30
                              ? fmin(4.0, fmax(0.25, first / second))
                              : 1.0;
    }
}

static int qc_compare_double_host(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

__global__ static void qc_cone_update_kernel(
    const HPRQPGPUCsrIndex *c_row_ptr,
    const HPRQPGPUCsrIndex *c_col_ind,
    const double *c_values,
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const double *constant,
    const double *offset,
    const double *x_hat,
    const double *kappa,
    const double *theta,
    const double *last_dual,
    double *dual,
    double *dual_bar,
    double *dual_delta,
    double *metric_energy,
    double sigma_d_s,
    const double *cone_metric_scale,
    double halpern_anchor,
    double halpern_reflection,
    int to_check,
    size_t p)
{
    size_t cone = (size_t)blockIdx.x * (size_t)blockDim.x +
                  (size_t)threadIdx.x;
    if (cone >= p) {
        return;
    }

    const HPRQPGPUCsrIndex qb = q_row_ptr[cone];
    const HPRQPGPUCsrIndex qe = q_row_ptr[cone + 1];
    const size_t start = 2 * cone + (size_t)qb;
    const double shape = theta[cone];
    const double sqrt_shape = sqrt(shape);
    const double metric = sigma_d_s * kappa[cone] *
        (cone_metric_scale != NULL ? cone_metric_scale[cone] : 1.0);
    const double old0 = dual[start];
    const double old1 = dual[start + 1];
    double linear = 0.0;

    for (HPRQPGPUCsrIndex ci = c_row_ptr[cone];
         ci < c_row_ptr[cone + 1]; ++ci) {
        linear += c_values[ci] * x_hat[c_col_ind[ci]];
    }

    double r0 = constant[cone] - (metric / shape) * old0;
    double r1 = -linear - offset[cone] - (metric * shape) * old1;
    double projected0 = sqrt_shape * r0;
    double projected1 = r1 / sqrt_shape;
    double tail_norm_sq = 0.0;
    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        const size_t tail = start + 2 + (size_t)(qi - qb);
        const double r = q_values[qi] * x_hat[q_col_ind[qi]] -
                         metric * dual[tail];
        tail_norm_sq += r * r;
    }
    double tail_scale;
    project_rotated_soc(&projected0, &projected1, sqrt(tail_norm_sq),
                        &tail_scale);
    projected0 /= sqrt_shape;
    projected1 *= sqrt_shape;

    const double bar0 = (projected0 - r0) / (metric / shape);
    const double bar1 = (projected1 - r1) / (metric * shape);
    if (to_check) {
        dual_bar[start] = bar0;
        dual_bar[start + 1] = bar1;
        dual_delta[start] = bar0 - old0;
        dual_delta[start + 1] = bar1 - old1;
    }
    dual[start] = halpern_reflection * (2.0 * bar0 - old0) +
                  halpern_anchor * last_dual[start];
    dual[start + 1] = halpern_reflection * (2.0 * bar1 - old1) +
                      halpern_anchor * last_dual[start + 1];

    for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
        const size_t tail = start + 2 + (size_t)(qi - qb);
        const double old = dual[tail];
        const double r = q_values[qi] * x_hat[q_col_ind[qi]] -
                         metric * old;
        const double projected = tail_scale * r;
        const double bar = (projected - r) / metric;
        if (to_check) {
            dual_bar[tail] = bar;
            dual_delta[tail] = bar - old;
        }
        dual[tail] = halpern_reflection * (2.0 * bar - old) +
                     halpern_anchor * last_dual[tail];
    }
    if (to_check) {
        double energy = dual_delta[start] * dual_delta[start] / shape +
                        shape * dual_delta[start + 1] *
                            dual_delta[start + 1];
        for (HPRQPGPUCsrIndex qi = qb; qi < qe; ++qi) {
            const double value =
                dual_delta[start + 2U + (size_t)(qi - qb)];
            energy += value * value;
        }
        metric_energy[cone] = kappa[cone] * energy;
    }
}

__device__ static double qc_csr_row_dot(
    const HPRQPGPUCsrIndex *row_ptr,
    const HPRQPGPUCsrIndex *col_ind,
    const double *values,
    const double *input,
    size_t row)
{
    double sum = 0.0;
    for (HPRQPGPUCsrIndex index = row_ptr[row];
         index < row_ptr[row + 1U]; ++index) {
        sum += values[index] * input[col_ind[index]];
    }
    return sum;
}

__global__ static void qc_xw_update_kernel(
    const double *c,
    const double *lower,
    const double *upper,
    const double *Qw,
    const double *ATy,
    const double *BTs,
    const double *last_x,
    const double *last_w,
    double *x,
    double *x_bar,
    double *x_hat,
    double *z_bar,
    double *dx,
    double *w,
    double *w_bar,
    double *dw,
    double sigma,
    double w_gamma,
    double w_inv,
    const double *w_gamma_diag,
    const double *w_inv_diag,
    double halpern_anchor,
    double halpern_reflection,
    int has_A,
    int to_check,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    if (i >= n) return;
    const double old_x = x[i];
    const double old_w = w[i];
    const double stationarity = -Qw[i] + (has_A ? ATy[i] : 0.0) +
                                BTs[i] - c[i];
    const double raw = old_x + sigma * stationarity;
    const double bar_x = fmin(fmax(raw, lower[i]), upper[i]);
    const double reflected_x = 2.0 * bar_x - old_x;
    const double gamma_i = w_gamma_diag != NULL ? w_gamma_diag[i] : w_gamma;
    const double inv_i = w_inv_diag != NULL ? w_inv_diag[i] : w_inv;
    const double bar_w = (gamma_i * old_w + reflected_x) * inv_i;

    x_hat[i] = reflected_x;
    if (to_check) {
        x_bar[i] = bar_x;
        z_bar[i] = (bar_x - raw) / sigma;
        w_bar[i] = bar_w;
        dx[i] = bar_x - old_x;
        dw[i] = bar_w - old_w;
    }
    x[i] = halpern_reflection * reflected_x + halpern_anchor * last_x[i];
    w[i] = halpern_reflection * (2.0 * bar_w - old_w) +
           halpern_anchor * last_w[i];
}

/*
 * Preserve the ordinary QCQP arithmetic while avoiding materialization of
 * Q*w and B^*s when Q has at most one stored diagonal entry per row and the
 * transpose of B has short rows.  Both row sums use the same left-to-right
 * scalar accumulation as the standalone customized SpMV/gather kernels.
 * Qw and BTs are still written at check iterations because the metric and
 * residual calculations consume those exact pre-update products.
 */
__global__ static void qc_xw_update_fused_q_bt_kernel(
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const HPRQPGPUCsrIndex *bt_row_ptr,
    const HPRQPGPUCsrIndex *bt_dual_ind,
    const double *bt_values,
    const double *qc_dual,
    const double *c,
    const double *lower,
    const double *upper,
    const double *ATy,
    const double *last_x,
    const double *last_w,
    double *Qw,
    double *BTs,
    double *x,
    double *x_bar,
    double *x_hat,
    double *z_bar,
    double *dx,
    double *w,
    double *w_bar,
    double *dw,
    double sigma,
    double w_gamma,
    double w_inv,
    const double *w_gamma_diag,
    const double *w_inv_diag,
    double halpern_anchor,
    double halpern_reflection,
    int has_A,
    int to_check,
    size_t n)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    double q_w;
    double bt_s = 0.0;
    double old_x;
    double old_w;
    double stationarity;
    double raw;
    double bar_x;
    double reflected_x;
    double bar_w;
    if (i >= n) return;

    q_w = qc_csr_row_dot(q_row_ptr, q_col_ind, q_values, w, i);
    for (HPRQPGPUCsrIndex index = bt_row_ptr[i];
         index < bt_row_ptr[i + 1U]; ++index) {
        bt_s += bt_values[index] * qc_dual[bt_dual_ind[index]];
    }
    old_x = x[i];
    old_w = w[i];
    stationarity = -q_w + (has_A ? ATy[i] : 0.0) + bt_s - c[i];
    raw = old_x + sigma * stationarity;
    bar_x = fmin(fmax(raw, lower[i]), upper[i]);
    reflected_x = 2.0 * bar_x - old_x;
    {
        const double gamma_i =
            w_gamma_diag != NULL ? w_gamma_diag[i] : w_gamma;
        const double inv_i = w_inv_diag != NULL ? w_inv_diag[i] : w_inv;
        bar_w = (gamma_i * old_w + reflected_x) * inv_i;
    }

    x_hat[i] = reflected_x;
    if (to_check) {
        Qw[i] = q_w;
        BTs[i] = bt_s;
        x_bar[i] = bar_x;
        z_bar[i] = (bar_x - raw) / sigma;
        w_bar[i] = bar_w;
        dx[i] = bar_x - old_x;
        dw[i] = bar_w - old_w;
    }
    x[i] = halpern_reflection * reflected_x + halpern_anchor * last_x[i];
    w[i] = halpern_reflection * (2.0 * bar_w - old_w) +
           halpern_anchor * last_w[i];
}

__global__ static void qc_xw_update_fused_q_at_kernel(
    const HPRQPGPUCsrIndex *q_row_ptr,
    const HPRQPGPUCsrIndex *q_col_ind,
    const double *q_values,
    const HPRQPGPUCsrIndex *at_row_ptr,
    const HPRQPGPUCsrIndex *at_col_ind,
    const double *at_values,
    const double *c,
    const double *lower,
    const double *upper,
    const double *y,
    const double *BTs,
    const double *last_x,
    const double *last_w,
    double *Qw,
    double *ATy,
    double *x,
    double *x_bar,
    double *x_hat,
    double *z_bar,
    double *w,
    double *w_bar,
    double sigma,
    double w_gamma,
    double w_inv,
    const double *w_gamma_diag,
    const double *w_inv_diag,
    double halpern_anchor,
    double halpern_reflection,
    int has_Q,
    int has_A,
    size_t n)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    double q_w;
    double at_y;
    double old_x;
    double old_w;
    double stationarity;
    double raw;
    double bar_x;
    double reflected_x;
    double bar_w;
    if (i >= n) return;

    q_w = has_Q ? qc_csr_row_dot(q_row_ptr, q_col_ind, q_values, w, i)
                 : 0.0;
    at_y = has_A ? qc_csr_row_dot(at_row_ptr, at_col_ind, at_values, y, i)
                 : 0.0;
    old_x = x[i];
    old_w = w[i];
    stationarity = -q_w + at_y + BTs[i] - c[i];
    raw = old_x + sigma * stationarity;
    bar_x = fmin(fmax(raw, lower[i]), upper[i]);
    reflected_x = 2.0 * bar_x - old_x;
    {
        const double gamma_i =
            w_gamma_diag != NULL ? w_gamma_diag[i] : w_gamma;
        const double inv_i = w_inv_diag != NULL ? w_inv_diag[i] : w_inv;
        bar_w = (gamma_i * old_w + reflected_x) * inv_i;
    }

    x_hat[i] = reflected_x;
    x[i] = halpern_reflection * reflected_x + halpern_anchor * last_x[i];
    w[i] = halpern_reflection * (2.0 * bar_w - old_w) +
           halpern_anchor * last_w[i];
}

__global__ static void qc_y_update_kernel(
    const double *Ax,
    const double *lower,
    const double *upper,
    const double *last_y,
    double *y,
    double *y_bar,
    double *dy,
    double *projection,
    double rho,
    double inv_rho,
    double halpern_anchor,
    double halpern_reflection,
    int to_check,
    size_t m)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    if (i >= m) return;
    const double old = y[i];
    const double raw = Ax[i] - rho * old;
    const double projected = fmin(fmax(raw, lower[i]), upper[i]);
    const double bar = (projected - raw) * inv_rho;
    if (to_check) {
        projection[i] = projected;
        y_bar[i] = bar;
        dy[i] = bar - old;
    }
    y[i] = halpern_reflection * (2.0 * bar - old) +
           halpern_anchor * last_y[i];
}

__global__ static void qc_y_update_fused_a_kernel(
    const HPRQPGPUCsrIndex *a_row_ptr,
    const HPRQPGPUCsrIndex *a_col_ind,
    const double *a_values,
    const double *x_hat,
    const double *lower,
    const double *upper,
    const double *last_y,
    double *Ax,
    double *y,
    double *y_bar,
    double *dy,
    double *projection,
    double rho,
    double inv_rho,
    double halpern_anchor,
    double halpern_reflection,
    int to_check,
    size_t m)
{
    const size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
                     (size_t)threadIdx.x;
    double ax;
    double old;
    double raw;
    double projected;
    double bar;
    if (i >= m) return;

    ax = qc_csr_row_dot(a_row_ptr, a_col_ind, a_values, x_hat, i);
    Ax[i] = ax;
    old = y[i];
    raw = ax - rho * old;
    projected = fmin(fmax(raw, lower[i]), upper[i]);
    bar = (projected - raw) * inv_rho;
    if (to_check) {
        projection[i] = projected;
        y_bar[i] = bar;
        dy[i] = bar - old;
    }
    y[i] = halpern_reflection * (2.0 * bar - old) +
           halpern_anchor * last_y[i];
}

__global__ static void qc_check_differences_kernel(
    const double *Qw,
    const double *Qw_bar,
    const double *ATy,
    const double *ATy_bar,
    const double *BTs,
    const double *BTs_bar,
    double *dQw,
    double *ATdy,
    double *BTds,
    int has_A,
    size_t n)
{
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x +
               (size_t)threadIdx.x;
    if (i >= n) return;
    dQw[i] = Qw_bar[i] - Qw[i];
    ATdy[i] = has_A ? ATy_bar[i] - ATy[i] : 0.0;
    BTds[i] = BTs_bar[i] - BTs[i];
}

static HPRQPStatus qc_zero_vector(double *values, size_t length)
{
    if (length == 0U) return HPRQP_STATUS_OK;
    return map_cuda_status(cudaMemset(values, 0, length * sizeof(double)));
}

static HPRQPStatus qc_launch_scale_copy(double *output,
                                        const double *input,
                                        double scale,
                                        size_t length,
                                        unsigned int threads)
{
    if (length == 0U) return HPRQP_STATUS_OK;
    const unsigned int blocks = (unsigned int)((length + threads - 1U) /
                                                threads);
    qc_scale_copy_kernel<<<blocks, threads>>>(output, input, scale, length);
    return map_cuda_status(cudaGetLastError());
}

static HPRQPStatus qc_launch_probe(double *values,
                                   size_t length,
                                   unsigned int threads)
{
    if (length == 0U) return HPRQP_STATUS_OK;
    const unsigned int blocks = (unsigned int)((length + threads - 1U) /
                                                threads);
    qc_probe_kernel<<<blocks, threads>>>(values, length);
    return map_cuda_status(cudaGetLastError());
}

static HPRQPStatus qc_apply_normalized_gram(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem,
    const double *input,
    double *output)
{
    HPRQPStatus status = qc_zero_vector(output, workspace->n);
    if (status != HPRQP_STATUS_OK) return status;
    qc_apply_normalized_gram_kernel<<<workspace->blocks_p,
                                      workspace->threads_p>>>(
        problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
        problem->qc_linear.values, problem->qc_factor.row_ptr,
        problem->qc_factor.col_ind, problem->qc_factor.values,
        workspace->qc_kappa.values, workspace->qc_theta.values,
        input, output, workspace->p);
    return map_cuda_status(cudaGetLastError());
}

/* Estimate lambda_max(F^*F), F=[A;B], without materializing F.  During the
 * iteration qc_kappa=qc_theta=1, so qc_apply_normalized_gram applies B^*B. */
static HPRQPStatus qc_power_scalar_f(HPRQPGPUWorkspace *workspace,
                                     const HPRQPGPUProblem *problem,
                                     const HPRQPParams *params,
                                     double *lambda_f)
{
    const size_t max_iterations = 500U;
    const double tolerance = 1e-4;
    const double one = 1.0;
    cublasHandle_t handle = gpu_blas_handle(workspace);
    double eigenvalue = 0.0;
    double residual_ratio = INFINITY;
    int converged = 0;
    HPRQPStatus status;

    if (lambda_f == NULL || handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    status = qc_launch_probe(workspace->tempv.values, workspace->n,
                             workspace->threads_n);
    if (status != HPRQP_STATUS_OK) return status;

    for (size_t iteration = 0U; iteration < max_iterations; ++iteration) {
        double vector_norm;
        double image_norm;
        double residual_norm;

        status = gpu_raw_norm2(handle, workspace->n,
                               workspace->tempv.values, &vector_norm);
        if (status != HPRQP_STATUS_OK) return status;
        if (vector_norm <= 1e-16) break;
        status = qc_launch_scale_copy(workspace->x_bar.values,
                                      workspace->tempv.values,
                                      1.0 / vector_norm, workspace->n,
                                      workspace->threads_n);
        if (status != HPRQP_STATUS_OK) return status;

        if (!workspace->empty_A) {
            status = gpu_spmv_apply_a_x_bar(workspace, problem);
            if (status != HPRQP_STATUS_OK) return status;
            status = qc_launch_scale_copy(workspace->y_bar.values,
                                          workspace->Ax.values, 1.0,
                                          workspace->m,
                                          workspace->threads_m);
            if (status != HPRQP_STATUS_OK) return status;
            status = gpu_spmv_apply_at_y_bar(workspace, problem);
            if (status != HPRQP_STATUS_OK) return status;
        } else {
            status = qc_zero_vector(workspace->ATy_bar.values,
                                    workspace->n);
            if (status != HPRQP_STATUS_OK) return status;
        }
        status = qc_apply_normalized_gram(workspace, problem,
                                          workspace->x_bar.values,
                                          workspace->BTs.values);
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(workspace->Qx.values,
                                            workspace->ATy_bar.values,
                                            workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cublas_status(cublasDaxpy(
            handle, (int)workspace->n, &one, workspace->BTs.values, 1,
            workspace->Qx.values, 1));
        if (status != HPRQP_STATUS_OK) return status;

        status = gpu_raw_dot(handle, workspace->n, workspace->x_bar.values,
                             workspace->Qx.values, &eigenvalue);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->n, workspace->Qx.values,
                               &image_norm);
        if (status != HPRQP_STATUS_OK) return status;
        qc_eigen_residual_kernel<<<workspace->blocks_n,
                                   workspace->threads_n>>>(
            workspace->dx.values, workspace->Qx.values,
            workspace->x_bar.values, eigenvalue, workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->n, workspace->dx.values,
                               &residual_norm);
        if (status != HPRQP_STATUS_OK) return status;
        residual_ratio = residual_norm /
                         (image_norm + fabs(eigenvalue) + 1e-16);
        if (residual_ratio < tolerance) {
            converged = 1;
            break;
        }
        status = map_cuda_status(cudaMemcpy(workspace->tempv.values,
                                            workspace->Qx.values,
                                            workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
    }
    if (!(isfinite(eigenvalue) && eigenvalue > 0.0)) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    *lambda_f = params->eig_factor * eigenvalue;
    if (params->verbose) {
        printf("QCQP scalar-F power: converged=%d residual=%.6e "
               "ritz=%.12g\n", converged, residual_ratio, eigenvalue);
    }

    (void)qc_zero_vector(workspace->tempv.values, workspace->n);
    (void)qc_zero_vector(workspace->x_bar.values, workspace->n);
    (void)qc_zero_vector(workspace->Qx.values, workspace->n);
    (void)qc_zero_vector(workspace->BTs.values, workspace->n);
    (void)qc_zero_vector(workspace->ATy_bar.values, workspace->n);
    (void)qc_zero_vector(workspace->Ax.values, workspace->m);
    (void)qc_zero_vector(workspace->y_bar.values, workspace->m);
    (void)qc_zero_vector(workspace->dx.values, workspace->n);
    return HPRQP_STATUS_OK;
}

/* Estimate lambda_max(B^*B) without materializing B.  During this power
 * iteration qc_kappa=qc_theta=1, so qc_apply_normalized_gram applies B^*B. */
static HPRQPStatus qc_power_scalar_b(HPRQPGPUWorkspace *workspace,
                                     const HPRQPGPUProblem *problem,
                                     const HPRQPParams *params,
                                     double *lambda_b)
{
    const size_t max_iterations = 500U;
    const double tolerance = 1e-4;
    cublasHandle_t handle = gpu_blas_handle(workspace);
    double eigenvalue = 0.0;
    double residual_ratio = INFINITY;
    int converged = 0;
    HPRQPStatus status;

    if (lambda_b == NULL || handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    status = qc_launch_probe(workspace->tempv.values, workspace->n,
                             workspace->threads_n);
    if (status != HPRQP_STATUS_OK) return status;

    for (size_t iteration = 0U; iteration < max_iterations; ++iteration) {
        double vector_norm;
        double image_norm;
        double residual_norm;

        status = gpu_raw_norm2(handle, workspace->n,
                               workspace->tempv.values, &vector_norm);
        if (status != HPRQP_STATUS_OK) return status;
        if (vector_norm <= 1e-16) break;
        status = qc_launch_scale_copy(workspace->x_bar.values,
                                      workspace->tempv.values,
                                      1.0 / vector_norm, workspace->n,
                                      workspace->threads_n);
        if (status != HPRQP_STATUS_OK) return status;
        status = qc_apply_normalized_gram(workspace, problem,
                                          workspace->x_bar.values,
                                          workspace->BTs.values);
        if (status != HPRQP_STATUS_OK) return status;

        status = gpu_raw_dot(handle, workspace->n, workspace->x_bar.values,
                             workspace->BTs.values, &eigenvalue);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->n, workspace->BTs.values,
                               &image_norm);
        if (status != HPRQP_STATUS_OK) return status;
        qc_eigen_residual_kernel<<<workspace->blocks_n,
                                   workspace->threads_n>>>(
            workspace->dx.values, workspace->BTs.values,
            workspace->x_bar.values, eigenvalue, workspace->n);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->n, workspace->dx.values,
                               &residual_norm);
        if (status != HPRQP_STATUS_OK) return status;
        residual_ratio = residual_norm /
                         (image_norm + fabs(eigenvalue) + 1e-16);
        if (residual_ratio < tolerance) {
            converged = 1;
            break;
        }
        status = map_cuda_status(cudaMemcpy(workspace->tempv.values,
                                            workspace->BTs.values,
                                            workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
    }
    if (!(isfinite(eigenvalue) && eigenvalue > 0.0)) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    *lambda_b = params->eig_factor * eigenvalue;
    if (params->verbose) {
        printf("QCQP scalar-B power: converged=%d residual=%.6e "
               "ritz=%.12g lambda_B=%.12g\n",
               converged, residual_ratio, eigenvalue, *lambda_b);
    }

    (void)qc_zero_vector(workspace->tempv.values, workspace->n);
    (void)qc_zero_vector(workspace->x_bar.values, workspace->n);
    (void)qc_zero_vector(workspace->BTs.values, workspace->n);
    (void)qc_zero_vector(workspace->dx.values, workspace->n);
    return HPRQP_STATUS_OK;
}

/* Estimate the normalized coupling between F=[A;B] and Q by applying
 * F Q F^* on the product dual space.  During this setup power iteration,
 * qc_kappa=qc_theta=1, hence the whitened B kernels apply the raw B/B^*.
 * The result is
 *
 *   zeta_FQ^2 = lambda_max(F Q F^*) / (lambda_F lambda_Q).
 */
static HPRQPStatus qc_power_coupling_fq(HPRQPGPUWorkspace *workspace,
                                        const HPRQPGPUProblem *problem,
                                        const HPRQPParams *params,
                                        double *coupling)
{
    const size_t max_iterations = 500U;
    const double tolerance = 1e-4;
    const double one = 1.0;
    cublasHandle_t handle = gpu_blas_handle(workspace);
    const unsigned int cone_blocks = (unsigned int)(
        (workspace->qc_cone_dim + workspace->threads_p - 1U) /
        workspace->threads_p);
    double eigenvalue = 0.0;
    double residual_ratio = INFINITY;
    int converged = 0;
    HPRQPStatus status;

    if (coupling == NULL || handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (problem->Q.nnz == 0U || !(workspace->lambda_max_Q > 0.0)) {
        *coupling = 0.0;
        return HPRQP_STATUS_OK;
    }
    if (!(workspace->lambda_F > 0.0)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    *coupling = 1.0;
    if (!workspace->empty_A) {
        status = qc_launch_probe(workspace->Ax.values, workspace->m,
                                 workspace->threads_m);
        if (status != HPRQP_STATUS_OK) return status;
    }
    status = qc_launch_probe(workspace->qc_dual_delta.values,
                             workspace->qc_cone_dim,
                             workspace->threads_p);
    if (status != HPRQP_STATUS_OK) return status;

    for (size_t iteration = 0U; iteration < max_iterations; ++iteration) {
        double norm_y = 0.0;
        double norm_s = 0.0;
        double image_y = 0.0;
        double image_s = 0.0;
        double residual_y = 0.0;
        double residual_s = 0.0;
        double dot_y = 0.0;
        double dot_s = 0.0;
        double vector_norm;
        double image_norm;
        double residual_norm;

        if (!workspace->empty_A) {
            status = gpu_raw_norm2(handle, workspace->m,
                                   workspace->Ax.values, &norm_y);
            if (status != HPRQP_STATUS_OK) return status;
        }
        status = gpu_raw_norm2(handle, workspace->qc_cone_dim,
                               workspace->qc_dual_delta.values, &norm_s);
        if (status != HPRQP_STATUS_OK) return status;
        vector_norm = hypot(norm_y, norm_s);
        if (vector_norm <= 1e-16) {
            eigenvalue = 0.0;
            converged = 1;
            break;
        }
        if (!workspace->empty_A) {
            status = qc_launch_scale_copy(workspace->y_bar.values,
                                          workspace->Ax.values,
                                          1.0 / vector_norm,
                                          workspace->m,
                                          workspace->threads_m);
            if (status != HPRQP_STATUS_OK) return status;
            status = gpu_spmv_apply_at_y_bar(workspace, problem);
            if (status != HPRQP_STATUS_OK) return status;
        } else {
            status = qc_zero_vector(workspace->ATy_bar.values,
                                    workspace->n);
            if (status != HPRQP_STATUS_OK) return status;
        }
        status = qc_launch_scale_copy(workspace->qc_dual_bar.values,
                                      workspace->qc_dual_delta.values,
                                      1.0 / vector_norm,
                                      workspace->qc_cone_dim,
                                      workspace->threads_p);
        if (status != HPRQP_STATUS_OK) return status;

        status = qc_zero_vector(workspace->BTs.values, workspace->n);
        if (status != HPRQP_STATUS_OK) return status;
        qc_apply_whitened_bt_kernel<<<workspace->blocks_p,
                                      workspace->threads_p>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            workspace->qc_kappa.values, workspace->qc_theta.values,
            workspace->qc_dual_bar.values, workspace->BTs.values,
            workspace->p);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(workspace->x_bar.values,
                                            workspace->ATy_bar.values,
                                            workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cublas_status(cublasDaxpy(
            handle, (int)workspace->n, &one, workspace->BTs.values, 1,
            workspace->x_bar.values, 1));
        if (status != HPRQP_STATUS_OK) return status;

        status = gpu_spmv_apply_q_x_bar(workspace, problem);
        if (status != HPRQP_STATUS_OK) return status;
        if (!workspace->empty_A) {
            status = map_cuda_status(cudaMemcpy(workspace->x_bar.values,
                                                workspace->Qx.values,
                                                workspace->n * sizeof(double),
                                                cudaMemcpyDeviceToDevice));
            if (status != HPRQP_STATUS_OK) return status;
            status = gpu_spmv_apply_a_x_bar(workspace, problem);
            if (status != HPRQP_STATUS_OK) return status;
        }
        qc_apply_whitened_b_kernel<<<workspace->blocks_p,
                                     workspace->threads_p>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            workspace->qc_kappa.values, workspace->qc_theta.values,
            workspace->Qx.values, workspace->qc_dual_delta.values,
            workspace->p);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;

        if (!workspace->empty_A) {
            status = gpu_raw_dot(handle, workspace->m,
                                 workspace->y_bar.values,
                                 workspace->Ax.values, &dot_y);
            if (status != HPRQP_STATUS_OK) return status;
            status = gpu_raw_norm2(handle, workspace->m,
                                   workspace->Ax.values, &image_y);
            if (status != HPRQP_STATUS_OK) return status;
        }
        status = gpu_raw_dot(handle, workspace->qc_cone_dim,
                             workspace->qc_dual_bar.values,
                             workspace->qc_dual_delta.values, &dot_s);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->qc_cone_dim,
                               workspace->qc_dual_delta.values, &image_s);
        if (status != HPRQP_STATUS_OK) return status;
        eigenvalue = dot_y + dot_s;
        image_norm = hypot(image_y, image_s);

        if (!workspace->empty_A) {
            qc_eigen_residual_kernel<<<workspace->blocks_m,
                                       workspace->threads_m>>>(
                workspace->y.values, workspace->Ax.values,
                workspace->y_bar.values, eigenvalue, workspace->m);
            status = map_cuda_status(cudaGetLastError());
            if (status != HPRQP_STATUS_OK) return status;
            status = gpu_raw_norm2(handle, workspace->m,
                                   workspace->y.values, &residual_y);
            if (status != HPRQP_STATUS_OK) return status;
        }
        qc_eigen_residual_kernel<<<cone_blocks, workspace->threads_p>>>(
            workspace->qc_dual.values,
            workspace->qc_dual_delta.values,
            workspace->qc_dual_bar.values, eigenvalue,
            workspace->qc_cone_dim);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->qc_cone_dim,
                               workspace->qc_dual.values, &residual_s);
        if (status != HPRQP_STATUS_OK) return status;
        residual_norm = hypot(residual_y, residual_s);
        residual_ratio = residual_norm /
                         (image_norm + fabs(eigenvalue) + 1e-16);
        if (residual_ratio < tolerance) {
            converged = 1;
            break;
        }
    }

    if (converged) {
        const double ratio = params->eig_factor * fmax(eigenvalue, 0.0) /
                             (workspace->lambda_F *
                              workspace->lambda_max_Q);
        *coupling = sqrt(fmin(1.0, fmax(0.0, ratio)));
    }
    if (params->verbose) {
        printf("QCQP joint-F/Q power: converged=%d residual=%.6e "
               "ritz=%.12g zeta_FQ=%.12g\n",
               converged, residual_ratio, eigenvalue, *coupling);
    }

    (void)qc_zero_vector(workspace->y.values, workspace->m);
    (void)qc_zero_vector(workspace->y_bar.values, workspace->m);
    (void)qc_zero_vector(workspace->Ax.values, workspace->m);
    (void)qc_zero_vector(workspace->ATy_bar.values, workspace->n);
    (void)qc_zero_vector(workspace->qc_dual.values,
                         workspace->qc_cone_dim);
    (void)qc_zero_vector(workspace->qc_dual_bar.values,
                         workspace->qc_cone_dim);
    (void)qc_zero_vector(workspace->qc_dual_delta.values,
                         workspace->qc_cone_dim);
    (void)qc_zero_vector(workspace->BTs.values, workspace->n);
    (void)qc_zero_vector(workspace->x_bar.values, workspace->n);
    (void)qc_zero_vector(workspace->Qx.values, workspace->n);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus qc_power_coupling_ab(HPRQPGPUWorkspace *workspace,
                                        const HPRQPGPUProblem *problem,
                                        const HPRQPParams *params,
                                        double *coupling)
{
    const size_t max_iterations = 500U;
    const double tolerance = 1e-4;
    cublasHandle_t handle = gpu_blas_handle(workspace);
    double eigenvalue = 0.0;
    int converged = 0;
    HPRQPStatus status;

    if (coupling == NULL || handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->empty_A || workspace->m == 0U ||
        !(workspace->lambda_max_A > 0.0)) {
        *coupling = 0.0;
        return HPRQP_STATUS_OK;
    }

    *coupling = 1.0;
    status = qc_launch_probe(workspace->Ax.values, workspace->m,
                             workspace->threads_m);
    if (status != HPRQP_STATUS_OK) return status;

    for (size_t iteration = 0U; iteration < max_iterations; ++iteration) {
        double vector_norm;
        double image_norm;
        double residual_norm;

        status = gpu_raw_norm2(handle, workspace->m, workspace->Ax.values,
                               &vector_norm);
        if (status != HPRQP_STATUS_OK) return status;
        if (vector_norm <= 1e-16) {
            eigenvalue = 0.0;
            converged = 1;
            break;
        }
        status = qc_launch_scale_copy(workspace->y_bar.values,
                                      workspace->Ax.values,
                                      1.0 / vector_norm, workspace->m,
                                      workspace->threads_m);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_spmv_apply_at_y_bar(workspace, problem);
        if (status != HPRQP_STATUS_OK) return status;
        status = qc_apply_normalized_gram(workspace, problem,
                                          workspace->ATy_bar.values,
                                          workspace->tempv.values);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_spmv_apply_a_tempv(workspace, problem);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_dot(handle, workspace->m, workspace->y_bar.values,
                             workspace->Ax.values, &eigenvalue);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->m, workspace->Ax.values,
                               &image_norm);
        if (status != HPRQP_STATUS_OK) return status;

        qc_eigen_residual_kernel<<<workspace->blocks_m,
                                   workspace->threads_m>>>(
            workspace->y.values, workspace->Ax.values,
            workspace->y_bar.values, eigenvalue, workspace->m);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->m, workspace->y.values,
                               &residual_norm);
        if (status != HPRQP_STATUS_OK) return status;
        if (residual_norm /
                (image_norm + fabs(eigenvalue) + 1e-16) < tolerance) {
            converged = 1;
            break;
        }
    }

    if (converged) {
        const double ratio = params->eig_factor * fmax(eigenvalue, 0.0) /
                             workspace->lambda_max_A;
        *coupling = sqrt(fmin(1.0, fmax(0.0, ratio)));
    }

    (void)qc_zero_vector(workspace->y.values, workspace->m);
    (void)qc_zero_vector(workspace->y_bar.values, workspace->m);
    (void)qc_zero_vector(workspace->Ax.values, workspace->m);
    (void)qc_zero_vector(workspace->ATy_bar.values, workspace->n);
    (void)qc_zero_vector(workspace->tempv.values, workspace->n);
    return HPRQP_STATUS_OK;
}

static HPRQPStatus qc_power_coupling_bq(HPRQPGPUWorkspace *workspace,
                                        const HPRQPGPUProblem *problem,
                                        const HPRQPParams *params,
                                        double *coupling)
{
    const size_t max_iterations = 500U;
    const double tolerance = 1e-4;
    cublasHandle_t handle = gpu_blas_handle(workspace);
    const unsigned int cone_blocks = (unsigned int)(
        (workspace->qc_cone_dim + workspace->threads_p - 1U) /
        workspace->threads_p);
    double eigenvalue = 0.0;
    int converged = 0;
    HPRQPStatus status;

    if (coupling == NULL || handle == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (problem->Q.nnz == 0U || !(workspace->lambda_max_Q > 0.0)) {
        *coupling = 0.0;
        return HPRQP_STATUS_OK;
    }

    *coupling = 1.0;
    status = qc_launch_probe(workspace->qc_dual_delta.values,
                             workspace->qc_cone_dim,
                             workspace->threads_p);
    if (status != HPRQP_STATUS_OK) return status;

    for (size_t iteration = 0U; iteration < max_iterations; ++iteration) {
        double vector_norm;
        double image_norm;
        double residual_norm;

        status = gpu_raw_norm2(handle, workspace->qc_cone_dim,
                               workspace->qc_dual_delta.values,
                               &vector_norm);
        if (status != HPRQP_STATUS_OK) return status;
        if (vector_norm <= 1e-16) {
            eigenvalue = 0.0;
            converged = 1;
            break;
        }
        status = qc_launch_scale_copy(workspace->qc_dual_bar.values,
                                      workspace->qc_dual_delta.values,
                                      1.0 / vector_norm,
                                      workspace->qc_cone_dim,
                                      workspace->threads_p);
        if (status != HPRQP_STATUS_OK) return status;
        status = qc_zero_vector(workspace->BTs.values, workspace->n);
        if (status != HPRQP_STATUS_OK) return status;
        qc_apply_whitened_bt_kernel<<<workspace->blocks_p,
                                      workspace->threads_p>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            workspace->qc_kappa.values, workspace->qc_theta.values,
            workspace->qc_dual_bar.values, workspace->BTs.values,
            workspace->p);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaMemcpy(workspace->x_bar.values,
                                            workspace->BTs.values,
                                            workspace->n * sizeof(double),
                                            cudaMemcpyDeviceToDevice));
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_spmv_apply_q_x_bar(workspace, problem);
        if (status != HPRQP_STATUS_OK) return status;
        qc_apply_whitened_b_kernel<<<workspace->blocks_p,
                                     workspace->threads_p>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            workspace->qc_kappa.values, workspace->qc_theta.values,
            workspace->Qx.values, workspace->qc_dual_delta.values,
            workspace->p);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_dot(handle, workspace->qc_cone_dim,
                             workspace->qc_dual_bar.values,
                             workspace->qc_dual_delta.values, &eigenvalue);
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->qc_cone_dim,
                               workspace->qc_dual_delta.values,
                               &image_norm);
        if (status != HPRQP_STATUS_OK) return status;

        qc_eigen_residual_kernel<<<cone_blocks, workspace->threads_p>>>(
            workspace->qc_dual.values,
            workspace->qc_dual_delta.values,
            workspace->qc_dual_bar.values, eigenvalue,
            workspace->qc_cone_dim);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = gpu_raw_norm2(handle, workspace->qc_cone_dim,
                               workspace->qc_dual.values, &residual_norm);
        if (status != HPRQP_STATUS_OK) return status;
        if (residual_norm /
                (image_norm + fabs(eigenvalue) + 1e-16) < tolerance) {
            converged = 1;
            break;
        }
    }

    if (converged) {
        const double ratio = params->eig_factor * fmax(eigenvalue, 0.0) /
                             workspace->lambda_max_Q;
        *coupling = sqrt(fmin(1.0, fmax(0.0, ratio)));
    }

    (void)qc_zero_vector(workspace->qc_dual.values,
                         workspace->qc_cone_dim);
    (void)qc_zero_vector(workspace->qc_dual_bar.values,
                         workspace->qc_cone_dim);
    (void)qc_zero_vector(workspace->qc_dual_delta.values,
                         workspace->qc_cone_dim);
    (void)qc_zero_vector(workspace->BTs.values, workspace->n);
    (void)qc_zero_vector(workspace->x_bar.values, workspace->n);
    (void)qc_zero_vector(workspace->Qx.values, workspace->n);
    return HPRQP_STATUS_OK;
}

static void qc_release_bt_gather(HPRQPGPUWorkspace *workspace)
{
    if (workspace->qc_bt_row_ptr != NULL) {
        cudaFree(workspace->qc_bt_row_ptr);
        workspace->qc_bt_row_ptr = NULL;
    }
    if (workspace->qc_bt_dual_ind != NULL) {
        cudaFree(workspace->qc_bt_dual_ind);
        workspace->qc_bt_dual_ind = NULL;
    }
    if (workspace->qc_bt_values != NULL) {
        cudaFree(workspace->qc_bt_values);
        workspace->qc_bt_values = NULL;
    }
    workspace->qc_bt_nnz = 0U;
    workspace->qc_bt_short_rows = 0;
}

static HPRQPStatus qc_prepare_bt_gather(HPRQPGPUWorkspace *workspace,
                                        const HPRQPGPUProblem *problem,
                                        int verbose)
{
    const size_t nnz = problem->qc_linear.nnz + problem->qc_factor.nnz;
    HPRQPGPUCsrIndex *c_row_ptr = NULL;
    HPRQPGPUCsrIndex *c_col_ind = NULL;
    HPRQPGPUCsrIndex *q_row_ptr = NULL;
    HPRQPGPUCsrIndex *q_col_ind = NULL;
    double *c_values = NULL;
    double *q_values = NULL;
    size_t *counts = NULL;
    HPRQPGPUCsrIndex *row_ptr = NULL;
    HPRQPGPUCsrIndex *cursor = NULL;
    HPRQPGPUCsrIndex *dual_ind = NULL;
    double *values = NULL;
    size_t max_row_nnz = 0U;
    HPRQPStatus status = HPRQP_STATUS_OK;

    qc_release_bt_gather(workspace);
    if (problem->n == 0U || nnz == 0U || nnz > (size_t)INT_MAX ||
        problem->qc_cone_dim > (size_t)INT_MAX) {
        return HPRQP_STATUS_OK;
    }

    c_row_ptr = (HPRQPGPUCsrIndex *)malloc(
        (problem->p + 1U) * sizeof(*c_row_ptr));
    q_row_ptr = (HPRQPGPUCsrIndex *)malloc(
        (problem->p + 1U) * sizeof(*q_row_ptr));
    if (problem->qc_linear.nnz > 0U) {
        c_col_ind = (HPRQPGPUCsrIndex *)malloc(
            problem->qc_linear.nnz * sizeof(*c_col_ind));
        c_values = (double *)malloc(
            problem->qc_linear.nnz * sizeof(*c_values));
    }
    if (problem->qc_factor.nnz > 0U) {
        q_col_ind = (HPRQPGPUCsrIndex *)malloc(
            problem->qc_factor.nnz * sizeof(*q_col_ind));
        q_values = (double *)malloc(
            problem->qc_factor.nnz * sizeof(*q_values));
    }
    counts = (size_t *)calloc(problem->n, sizeof(*counts));
    row_ptr = (HPRQPGPUCsrIndex *)malloc(
        (problem->n + 1U) * sizeof(*row_ptr));
    cursor = (HPRQPGPUCsrIndex *)malloc(problem->n * sizeof(*cursor));
    dual_ind = (HPRQPGPUCsrIndex *)malloc(nnz * sizeof(*dual_ind));
    values = (double *)malloc(nnz * sizeof(*values));
    if (c_row_ptr == NULL || q_row_ptr == NULL ||
        (problem->qc_linear.nnz > 0U &&
         (c_col_ind == NULL || c_values == NULL)) ||
        (problem->qc_factor.nnz > 0U &&
         (q_col_ind == NULL || q_values == NULL)) ||
        counts == NULL || row_ptr == NULL || cursor == NULL ||
        dual_ind == NULL || values == NULL) {
        status = HPRQP_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

#define QC_COPY_TO_HOST(dst, src, bytes)                                      \
    do {                                                                       \
        if ((bytes) > 0U &&                                                    \
            cudaMemcpy((dst), (src), (bytes), cudaMemcpyDeviceToHost) !=       \
                cudaSuccess) {                                                 \
            status = HPRQP_STATUS_NOT_IMPLEMENTED;                             \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)
    QC_COPY_TO_HOST(c_row_ptr, problem->qc_linear.row_ptr,
                    (problem->p + 1U) * sizeof(*c_row_ptr));
    QC_COPY_TO_HOST(q_row_ptr, problem->qc_factor.row_ptr,
                    (problem->p + 1U) * sizeof(*q_row_ptr));
    QC_COPY_TO_HOST(c_col_ind, problem->qc_linear.col_ind,
                    problem->qc_linear.nnz * sizeof(*c_col_ind));
    QC_COPY_TO_HOST(q_col_ind, problem->qc_factor.col_ind,
                    problem->qc_factor.nnz * sizeof(*q_col_ind));
    QC_COPY_TO_HOST(c_values, problem->qc_linear.values,
                    problem->qc_linear.nnz * sizeof(*c_values));
    QC_COPY_TO_HOST(q_values, problem->qc_factor.values,
                    problem->qc_factor.nnz * sizeof(*q_values));
#undef QC_COPY_TO_HOST

    for (size_t k = 0; k < problem->qc_linear.nnz; ++k) {
        const HPRQPGPUCsrIndex col = c_col_ind[k];
        if (col < 0 || (size_t)col >= problem->n) {
            status = HPRQP_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        ++counts[col];
    }
    for (size_t k = 0; k < problem->qc_factor.nnz; ++k) {
        const HPRQPGPUCsrIndex col = q_col_ind[k];
        if (col < 0 || (size_t)col >= problem->n) {
            status = HPRQP_STATUS_INVALID_ARGUMENT;
            goto cleanup;
        }
        ++counts[col];
    }
    row_ptr[0] = 0;
    for (size_t row = 0; row < problem->n; ++row) {
        if (counts[row] > max_row_nnz) max_row_nnz = counts[row];
        row_ptr[row + 1U] = row_ptr[row] + (HPRQPGPUCsrIndex)counts[row];
        cursor[row] = row_ptr[row];
    }
    if (max_row_nnz > 32U || nnz / problem->n > 4U) {
        if (verbose) {
            printf("QCQP B^T path: atomic scatter (nnz=%zu, max row=%zu)\n",
                   nnz, max_row_nnz);
        }
        goto cleanup;
    }

    for (size_t cone = 0; cone < problem->p; ++cone) {
        const HPRQPGPUCsrIndex qb = q_row_ptr[cone];
        const size_t start = 2U * cone + (size_t)qb;
        for (HPRQPGPUCsrIndex k = c_row_ptr[cone];
             k < c_row_ptr[cone + 1U]; ++k) {
            const HPRQPGPUCsrIndex col = c_col_ind[k];
            const HPRQPGPUCsrIndex pos = cursor[col]++;
            dual_ind[pos] = (HPRQPGPUCsrIndex)(start + 1U);
            values[pos] = -c_values[k];
        }
        for (HPRQPGPUCsrIndex k = qb; k < q_row_ptr[cone + 1U]; ++k) {
            const HPRQPGPUCsrIndex col = q_col_ind[k];
            const HPRQPGPUCsrIndex pos = cursor[col]++;
            dual_ind[pos] = (HPRQPGPUCsrIndex)(
                start + 2U + (size_t)(k - qb));
            values[pos] = q_values[k];
        }
    }

    if (cudaMalloc((void **)&workspace->qc_bt_row_ptr,
                   (problem->n + 1U) * sizeof(*row_ptr)) != cudaSuccess ||
        cudaMalloc((void **)&workspace->qc_bt_dual_ind,
                   nnz * sizeof(*dual_ind)) != cudaSuccess ||
        cudaMalloc((void **)&workspace->qc_bt_values,
                   nnz * sizeof(*values)) != cudaSuccess ||
        cudaMemcpy(workspace->qc_bt_row_ptr, row_ptr,
                   (problem->n + 1U) * sizeof(*row_ptr),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(workspace->qc_bt_dual_ind, dual_ind,
                   nnz * sizeof(*dual_ind), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(workspace->qc_bt_values, values,
                   nnz * sizeof(*values), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        qc_release_bt_gather(workspace);
        status = HPRQP_STATUS_NOT_IMPLEMENTED;
        goto cleanup;
    }
    workspace->qc_bt_nnz = nnz;
    workspace->qc_bt_short_rows = 1;
    if (verbose) {
        printf("QCQP B^T path: short-row gather (nnz=%zu, max row=%zu)\n",
               nnz, max_row_nnz);
    }

cleanup:
    free(c_row_ptr);
    free(c_col_ind);
    free(q_row_ptr);
    free(q_col_ind);
    free(c_values);
    free(q_values);
    free(counts);
    free(row_ptr);
    free(cursor);
    free(dual_ind);
    free(values);
    return status;
}

HPRQPStatus gpu_qcqp_refresh_diag_bq_metric(
    HPRQPGPUWorkspace *workspace,
    double base_cone_metric,
    double base_w_metric)
{
    HPRQPStatus status;
    cublasHandle_t handle;
    if (workspace == NULL || !workspace->qcqp_diag_bq ||
        !(workspace->lambda_max_Q > 0.0) ||
        !(workspace->sigma > 0.0) ||
        !(base_cone_metric > 0.0) || !(base_w_metric > 0.0)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    qc_diag_bq_refresh_cone_kernel<<<workspace->blocks_p,
                                      workspace->threads_p>>>(
        workspace->qc_diag_bq_row_weight.values,
        workspace->qc_diag_bq_cone_metric.values,
        base_cone_metric, workspace->p);
    qc_diag_bq_refresh_w_kernel<<<workspace->blocks_n,
                                   workspace->threads_n>>>(
        workspace->qc_diag_bq_col_weight.values,
        workspace->qc_diag_bq_w_gamma.values,
        workspace->qc_diag_bq_w_inv.values,
        workspace->sigma * workspace->lambda_max_Q,
        base_w_metric, workspace->n);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) return status;
    handle = gpu_blas_handle(workspace);
    if (handle == NULL) return HPRQP_STATUS_INVALID_ARGUMENT;
    status = gpu_raw_max_abs(handle, workspace->p,
                             workspace->qc_diag_bq_cone_metric.values,
                             &workspace->d_s);
    if (status != HPRQP_STATUS_OK) return status;
    status = gpu_raw_max_abs(handle, workspace->n,
                             workspace->qc_diag_bq_w_gamma.values,
                             &workspace->parallel_w_gamma);
    if (status != HPRQP_STATUS_OK) return status;
    workspace->d_w = workspace->parallel_w_gamma /
                     (workspace->sigma * workspace->lambda_max_Q);
    workspace->parallel_w_inv =
        1.0 / (1.0 + workspace->parallel_w_gamma);
    workspace->qcqp_diag_bq_base_s = base_cone_metric;
    workspace->qcqp_diag_bq_base_w = base_w_metric;
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_qcqp_diag_bq_metric_energies(
    HPRQPGPUWorkspace *workspace,
    double *cone_energy,
    double *w_energy)
{
    HPRQPStatus status;
    cublasHandle_t handle;
    if (workspace == NULL || cone_energy == NULL || w_energy == NULL ||
        !workspace->qcqp_diag_bq || !(workspace->sigma > 0.0)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    qc_diag_bq_cone_energy_kernel<<<workspace->blocks_p,
                                     workspace->threads_p>>>(
        workspace->qc_diag_bq_cone_metric.values,
        workspace->qc_primal_violation.values,
        workspace->qc_diag_bq_metric_energy.values,
        workspace->p);
    qc_diag_bq_w_energy_kernel<<<workspace->blocks_n,
                                  workspace->threads_n>>>(
        workspace->qc_diag_bq_w_gamma.values,
        workspace->dw.values, workspace->dQw.values,
        1.0 / workspace->sigma,
        workspace->qc_diag_bq_work.values,
        workspace->n);
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) return status;
    handle = gpu_blas_handle(workspace);
    if (handle == NULL) return HPRQP_STATUS_INVALID_ARGUMENT;
    status = map_cublas_status(cublasDasum(
        handle, (int)workspace->p,
        workspace->qc_diag_bq_metric_energy.values, 1, cone_energy));
    if (status != HPRQP_STATUS_OK) return status;
    return map_cublas_status(cublasDasum(
        handle, (int)workspace->n,
        workspace->qc_diag_bq_work.values, 1, w_energy));
}

HPRQPStatus gpu_qcqp_setup_majorizer(HPRQPGPUWorkspace *workspace,
                                     const HPRQPGPUProblem *problem,
                                     const HPRQPParams *params)
{
    int q_is_zero;
    int use_joint_f;
    int use_scalar_b;
    int use_diag_bq;

    if (workspace == NULL || problem == NULL || params == NULL ||
        problem->p == 0 || workspace->p != problem->p) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    {
        HPRQPStatus status = qc_prepare_bt_gather(
            workspace, problem, params->verbose);
        if (status != HPRQP_STATUS_OK) return status;
    }
    q_is_zero = problem->Q.nnz == 0U &&
                workspace->lambda_max_Q == 0.0;
    use_joint_f =
        params->qc_majorizer_mode == HPRQP_QC_MAJORIZER_JOINT_F ||
        (params->qc_majorizer_mode == HPRQP_QC_MAJORIZER_AUTO &&
         q_is_zero && !workspace->empty_A);
    use_scalar_b =
        params->qc_majorizer_mode == HPRQP_QC_MAJORIZER_SCALAR_B;
    use_diag_bq =
        params->qc_majorizer_mode == HPRQP_QC_MAJORIZER_DIAG_BQ;
    workspace->qcqp_diag_bq = 0;
    if (use_diag_bq && (!workspace->Q_is_diag || q_is_zero)) {
        if (params->verbose) {
            fprintf(stderr,
                    "diag-BQ majorizer requires a nonzero diagonal Q\n");
        }
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    qc_fill_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
        workspace->qc_theta.values, 1.0, workspace->p);
    if (use_joint_f) {
        HPRQPStatus status;
        double zeta_fq = q_is_zero ? 0.0 : 1.0;
        qc_fill_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
            workspace->qc_kappa.values, 1.0, workspace->p);
        status = qc_power_scalar_f(workspace, problem, params,
                                   &workspace->lambda_F);
        if (status != HPRQP_STATUS_OK) return status;
        if (!q_is_zero && params->parallel_majorizer_mode !=
                              HPRQP_PARALLEL_MAJORIZER_WORST_CASE) {
            status = qc_power_coupling_fq(workspace, problem, params,
                                          &zeta_fq);
            if (status != HPRQP_STATUS_OK) return status;
        }
        qc_fill_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
            workspace->qc_kappa.values, workspace->lambda_F, workspace->p);
        workspace->qcqp_scalar_f = 1;
        workspace->zeta_AB = 0.0;
        /* In scalar-F mode these legacy slots carry zeta_FQ and tau_FQ. */
        workspace->zeta_AQ = zeta_fq;
        workspace->zeta_BQ = 0.0;
        workspace->tau_AB = 1.0;
        workspace->tau_AQ = parallel_select_tau(
            params, workspace->lambda_F, workspace->lambda_max_Q);
        workspace->tau = workspace->tau_AQ;
        workspace->tau_BQ = 1.0;
        workspace->d_y = 1.0 + zeta_fq * workspace->tau_AQ;
        workspace->d_s = workspace->d_y;
        workspace->d_w = 1.0 + zeta_fq / workspace->tau_AQ;
        workspace->parallel_y_rho = workspace->empty_A
                                        ? 0.0
                                        : workspace->sigma *
                                              workspace->d_y *
                                              workspace->lambda_F;
        workspace->parallel_y_inv = workspace->empty_A
                                        ? 0.0
                                        : 1.0 /
                                              workspace->parallel_y_rho;
        workspace->parallel_w_gamma = workspace->sigma *
                                      workspace->d_w *
                                      workspace->lambda_max_Q;
        workspace->parallel_w_inv =
            1.0 / (1.0 + workspace->parallel_w_gamma);
        if (params->verbose) {
            printf("QCQP scalar-F majorizer: lambda_F=%.12g "
                   "zeta_FQ=%.12g tau_FQ=%.12g "
                   "d_F=%.12g d_w=%.12g\n",
                   workspace->lambda_F, workspace->zeta_AQ,
                   workspace->tau_AQ, workspace->d_y,
                   workspace->d_w);
        }
        return map_cuda_status(cudaGetLastError());
    }
    workspace->qcqp_scalar_f = 0;
    workspace->lambda_F = 0.0;
    if (use_scalar_b) {
        double lambda_b = 0.0;
        HPRQPStatus status;
        qc_fill_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
            workspace->qc_kappa.values, 1.0, workspace->p);
        status = qc_power_scalar_b(workspace, problem, params, &lambda_b);
        if (status != HPRQP_STATUS_OK) return status;
        qc_fill_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
            workspace->qc_kappa.values, lambda_b, workspace->p);
        if (params->verbose) {
            printf("QCQP scalar-B majorizer: D_B=lambda_B I, "
                   "lambda_B=%.12g\n", lambda_b);
        }
    } else {
        if (cudaMemset(workspace->qc_incidence_sum.values, 0,
                       workspace->n * sizeof(double)) != cudaSuccess) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
        qc_incidence_sum_kernel<<<workspace->blocks_p,
                                  workspace->threads_p>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            workspace->qc_theta.values, workspace->qc_incidence_sum.values,
            workspace->p);
        qc_kappa_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            workspace->qc_theta.values, workspace->qc_incidence_sum.values,
            workspace->qc_kappa.values, workspace->p);
        if (cudaGetLastError() != cudaSuccess) {
            return HPRQP_STATUS_NOT_IMPLEMENTED;
        }
    }

    workspace->zeta_AB = workspace->empty_A ? 0.0 : 1.0;
    workspace->zeta_BQ = workspace->lambda_max_Q > 0.0 ? 1.0 : 0.0;
    if (params->parallel_majorizer_mode !=
        HPRQP_PARALLEL_MAJORIZER_WORST_CASE) {
        HPRQPStatus status = qc_power_coupling_ab(
            workspace, problem, params, &workspace->zeta_AB);
        if (status != HPRQP_STATUS_OK) return status;
        status = qc_power_coupling_bq(
            workspace, problem, params, &workspace->zeta_BQ);
        if (status != HPRQP_STATUS_OK) return status;
    }
    workspace->tau_AB = 1.0;
    workspace->tau_AQ = workspace->tau;
    workspace->tau_BQ = 1.0;
    {
        HPRQPStatus status;
        double d_y_aq;
        double d_w_aq;
        double dummy_y_rho;
        double dummy_y_inv;
        double dummy_w_gamma;
        double dummy_w_inv;
        status = parallel_compute_coefficients(
            workspace->m,
            workspace->sigma,
            workspace->lambda_max_A,
            workspace->lambda_max_Q,
            workspace->lambda_max_AQA,
            workspace->lambda_max_AQ2A,
            workspace->zeta_AQ,
            workspace->tau_AQ,
            workspace->parallel_majorizer_mode,
            &d_y_aq,
            &d_w_aq,
            &dummy_y_rho,
            &dummy_y_inv,
            &dummy_w_gamma,
            &dummy_w_inv);
        if (status != HPRQP_STATUS_OK) return status;
        workspace->d_y = d_y_aq +
                         workspace->zeta_AB * workspace->tau_AB;
        if (use_diag_bq) {
            workspace->d_w = d_w_aq;
            workspace->qcqp_diag_bq_base_w = d_w_aq;
        } else {
            workspace->d_w = d_w_aq +
                             workspace->zeta_BQ / workspace->tau_BQ;
        }
    }
    workspace->d_s = 1.0 + workspace->zeta_AB / workspace->tau_AB +
                     (use_diag_bq ? 0.0 :
                      workspace->zeta_BQ * workspace->tau_BQ);

    workspace->parallel_y_rho = workspace->m > 0
                                    ? workspace->sigma * workspace->d_y *
                                          workspace->lambda_max_A
                                    : 0.0;
    workspace->parallel_y_inv = workspace->m > 0
                                    ? 1.0 / workspace->parallel_y_rho
                                    : 0.0;
    if (use_diag_bq) {
        HPRQPStatus status;
        status = map_cuda_status(cudaMemset(
            workspace->qc_diag_bq_col_weight.values, 0,
            workspace->n * sizeof(double)));
        if (status != HPRQP_STATUS_OK) return status;
        qc_diag_bq_weights_kernel<<<workspace->blocks_p,
                                     workspace->threads_p>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            workspace->diag_Q.values, workspace->qc_kappa.values,
            1.0 / sqrt(workspace->lambda_max_Q),
            workspace->qc_diag_bq_row_weight.values,
            workspace->qc_diag_bq_col_weight.values,
            workspace->p);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        workspace->qcqp_diag_bq = 1;
        status = gpu_qcqp_refresh_diag_bq_metric(
            workspace, workspace->d_s, workspace->d_w);
        if (status != HPRQP_STATUS_OK) return status;
        if (params->verbose) {
            printf("QCQP diag-BQ majorizer: zeta_AB=%.12g "
                   "zeta_BQ(reference)=%.12g max_d_s=%.12g "
                   "max_d_w=%.12g\n",
                   workspace->zeta_AB, workspace->zeta_BQ,
                   workspace->d_s, workspace->d_w);
        }
    } else {
        workspace->parallel_w_gamma = workspace->sigma * workspace->d_w *
                                      workspace->lambda_max_Q;
        workspace->parallel_w_inv =
            1.0 / (1.0 + workspace->parallel_w_gamma);
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_qcqp_apply_bt(const HPRQPGPUWorkspace *workspace,
                              const HPRQPGPUProblem *problem,
                              const double *dual,
                              double *output)
{
    if (workspace == NULL || problem == NULL || dual == NULL || output == NULL ||
        problem->p == 0) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->qc_bt_short_rows) {
        qc_bt_gather_kernel<<<workspace->blocks_n, workspace->threads_n>>>(
            workspace->qc_bt_row_ptr, workspace->qc_bt_dual_ind,
            workspace->qc_bt_values, dual, output, problem->n);
        return map_cuda_status(cudaGetLastError());
    }
    if (cudaMemset(output, 0, problem->n * sizeof(double)) != cudaSuccess) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    qc_bt_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
        problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
        problem->qc_linear.values, problem->qc_factor.row_ptr,
        problem->qc_factor.col_ind, problem->qc_factor.values,
        dual, output, problem->p);
    return map_cuda_status(cudaGetLastError());
}

HPRQPStatus gpu_qcqp_compute_residual_vectors(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem)
{
    if (workspace == NULL || problem == NULL || problem->p == 0U ||
        workspace->p != problem->p) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    qc_residual_vectors_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
        problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
        problem->qc_linear.values, problem->qc_factor.row_ptr,
        problem->qc_factor.col_ind, problem->qc_factor.values,
        problem->qc_constant.values, problem->qc_offset.values,
        workspace->qc_cr_shape.values,
        workspace->x_bar.values,
        workspace->qc_dual_bar.values,
        workspace->qc_primal_violation.values,
        workspace->qc_bx_norm.values, workspace->qc_h_norm.values,
        workspace->qc_dual_objective.values,
        workspace->p);
    return map_cuda_status(cudaGetLastError());
}

HPRQPStatus gpu_qcqp_compute_metric_energy(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem,
    const double *dual_delta)
{
    if (workspace == NULL || problem == NULL || dual_delta == NULL ||
        problem->p == 0U || workspace->p != problem->p) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    qc_metric_energy_kernel<<<workspace->blocks_p, workspace->threads_p>>>(
        problem->qc_factor.row_ptr, workspace->qc_kappa.values,
        workspace->qc_theta.values, dual_delta,
        workspace->qc_primal_violation.values, workspace->p);
    return map_cuda_status(cudaGetLastError());
}

HPRQPStatus gpu_qcqp_displacement_theta_stats(
    HPRQPGPUWorkspace *workspace,
    const HPRQPGPUProblem *problem,
    const double *dual_delta,
    double *theta_min,
    double *theta_median,
    double *theta_geomean,
    double *theta_max)
{
    double *host_theta;
    double log_sum = 0.0;
    if (workspace == NULL || problem == NULL || dual_delta == NULL ||
        theta_min == NULL || theta_median == NULL ||
        theta_geomean == NULL || theta_max == NULL ||
        problem->p == 0U || workspace->p != problem->p) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    qc_displacement_theta_kernel<<<workspace->blocks_p,
                                   workspace->threads_p>>>(
        problem->qc_factor.row_ptr, dual_delta,
        workspace->qc_primal_violation.values, workspace->p);
    if (cudaGetLastError() != cudaSuccess) {
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    host_theta = (double *)malloc(workspace->p * sizeof(double));
    if (host_theta == NULL) return HPRQP_STATUS_NOT_IMPLEMENTED;
    if (cudaMemcpy(host_theta, workspace->qc_primal_violation.values,
                   workspace->p * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(host_theta);
        return HPRQP_STATUS_NOT_IMPLEMENTED;
    }
    qsort(host_theta, workspace->p, sizeof(double), qc_compare_double_host);
    for (size_t i = 0; i < workspace->p; ++i) {
        log_sum += log(host_theta[i]);
    }
    *theta_min = host_theta[0];
    *theta_median = workspace->p % 2U != 0U
                        ? host_theta[workspace->p / 2U]
                        : 0.5 * (host_theta[workspace->p / 2U - 1U] +
                                 host_theta[workspace->p / 2U]);
    *theta_geomean = exp(log_sum / (double)workspace->p);
    *theta_max = host_theta[workspace->p - 1U];
    free(host_theta);
    return HPRQP_STATUS_OK;
}

HPRQPStatus gpu_qcqp_iteration_step_parallel(HPRQPGPUWorkspace *workspace,
                                             const HPRQPGPUProblem *problem,
                                             size_t inner_iter)
{
    HPRQPStatus status;
    const double anchor = 1.0 / ((double)inner_iter + 2.0);
    const double reflection = 1.0 - anchor;
    int fuse_q_at_xw;
    int fuse_q_bt_xw;
    int compact_update_blocks;
    int use_spmv_graphs;
    int overlap_cone;
    unsigned int update_threads_n;
    unsigned int update_threads_m;
    unsigned int update_threads_p;
    unsigned int update_blocks_n;
    unsigned int update_blocks_m;
    unsigned int update_blocks_p;

    if (workspace == NULL || problem == NULL || problem->p == 0 ||
        workspace->method != HPRQP_METHOD_PARALLEL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    fuse_q_at_xw = !workspace->to_check &&
        (problem->Q.nnz == 0U ||
         (workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED &&
          workspace->custom_spmv_q_kernel == HPRQP_GPU_CUSTOM_SPMV_ROW)) &&
        (workspace->empty_A ||
         (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED &&
          workspace->custom_spmv_a_kernel == HPRQP_GPU_CUSTOM_SPMV_ROW));
    compact_update_blocks =
        problem->Q.nnz > 0U && workspace->Q_is_diag &&
        workspace->parallel_q_exact_single_diagonal &&
        workspace->spmv_mode_q == HPRQP_SPMV_MODE_CUSTOMIZED &&
        workspace->custom_spmv_q_kernel == HPRQP_GPU_CUSTOM_SPMV_ROW &&
        workspace->qc_bt_short_rows;
    use_spmv_graphs = compact_update_blocks && !workspace->to_check &&
        workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSPARSE;
    fuse_q_bt_xw = !fuse_q_at_xw && compact_update_blocks;
    update_threads_n = compact_update_blocks ? 128U : workspace->threads_n;
    update_threads_m = compact_update_blocks ? 128U : workspace->threads_m;
    update_threads_p = compact_update_blocks ? 128U : workspace->threads_p;
    update_blocks_n = (unsigned int)((workspace->n + update_threads_n - 1U) /
                                     update_threads_n);
    update_blocks_m = workspace->empty_A
        ? 0U
        : (unsigned int)((workspace->m + update_threads_m - 1U) /
                         update_threads_m);
    update_blocks_p = (unsigned int)((workspace->p + update_threads_p - 1U) /
                                     update_threads_p);
    overlap_cone = !workspace->empty_A && workspace->p >= 1024U &&
                   workspace->qc_cone_stream != NULL &&
                   workspace->qc_x_ready_event != NULL &&
                   workspace->qc_cone_done_event != NULL;
    if (!fuse_q_bt_xw) {
        status = gpu_qcqp_apply_bt(workspace, problem,
                                   workspace->qc_dual.values,
                                   workspace->BTs.values);
        if (status != HPRQP_STATUS_OK) return status;
    }

    if (fuse_q_at_xw) {
        qc_xw_update_fused_q_at_kernel<<<update_blocks_n,
                                          update_threads_n>>>(
            problem->Q.row_ptr, problem->Q.col_ind, problem->Q.values,
            problem->AT.row_ptr, problem->AT.col_ind, problem->AT.values,
            problem->c.values, problem->l.values, problem->u.values,
            workspace->y.values, workspace->BTs.values,
            workspace->last_x.values, workspace->last_w.values,
            workspace->Qw.values, workspace->ATy.values,
            workspace->x.values, workspace->x_bar.values,
            workspace->x_hat.values, workspace->z_bar.values,
            workspace->w.values, workspace->w_bar.values,
            workspace->sigma, workspace->parallel_w_gamma,
            workspace->parallel_w_inv,
            workspace->qcqp_diag_bq
                ? workspace->qc_diag_bq_w_gamma.values
                : NULL,
            workspace->qcqp_diag_bq
                ? workspace->qc_diag_bq_w_inv.values
                : NULL,
            anchor, reflection,
            problem->Q.nnz > 0U, !workspace->empty_A, workspace->n);
    } else if (fuse_q_bt_xw) {
        if (!workspace->empty_A) {
            status = use_spmv_graphs
                ? gpu_spmv_apply_at_y_graph(workspace, problem)
                : gpu_spmv_apply_at_y(workspace, problem);
            if (status != HPRQP_STATUS_OK) return status;
        } else {
            status = map_cuda_status(cudaMemset(
                workspace->ATy.values, 0,
                workspace->n * sizeof(double)));
            if (status != HPRQP_STATUS_OK) return status;
        }
        qc_xw_update_fused_q_bt_kernel<<<update_blocks_n,
                                         update_threads_n>>>(
            problem->Q.row_ptr, problem->Q.col_ind, problem->Q.values,
            workspace->qc_bt_row_ptr, workspace->qc_bt_dual_ind,
            workspace->qc_bt_values, workspace->qc_dual.values,
            problem->c.values, problem->l.values, problem->u.values,
            workspace->ATy.values, workspace->last_x.values,
            workspace->last_w.values, workspace->Qw.values,
            workspace->BTs.values, workspace->x.values,
            workspace->x_bar.values, workspace->x_hat.values,
            workspace->z_bar.values, workspace->dx.values,
            workspace->w.values, workspace->w_bar.values,
            workspace->dw.values, workspace->sigma,
            workspace->parallel_w_gamma, workspace->parallel_w_inv,
            workspace->qcqp_diag_bq
                ? workspace->qc_diag_bq_w_gamma.values
                : NULL,
            workspace->qcqp_diag_bq
                ? workspace->qc_diag_bq_w_inv.values
                : NULL,
            anchor, reflection, !workspace->empty_A,
            workspace->to_check, workspace->n);
    } else {
        status = problem->Q.nnz > 0U
                     ? gpu_spmv_apply_q_w(workspace, problem)
                     : map_cuda_status(cudaMemset(
                           workspace->Qw.values, 0,
                           workspace->n * sizeof(double)));
        if (status != HPRQP_STATUS_OK) return status;
        if (!workspace->empty_A) {
            status = gpu_spmv_apply_at_y(workspace, problem);
            if (status != HPRQP_STATUS_OK) return status;
        } else {
            status = map_cuda_status(cudaMemset(
                workspace->ATy.values, 0,
                workspace->n * sizeof(double)));
            if (status != HPRQP_STATUS_OK) return status;
        }
        qc_xw_update_kernel<<<update_blocks_n, update_threads_n>>>(
            problem->c.values, problem->l.values, problem->u.values,
            workspace->Qw.values, workspace->ATy.values,
            workspace->BTs.values, workspace->last_x.values,
            workspace->last_w.values, workspace->x.values,
            workspace->x_bar.values, workspace->x_hat.values,
            workspace->z_bar.values, workspace->dx.values,
            workspace->w.values, workspace->w_bar.values,
            workspace->dw.values, workspace->sigma,
            workspace->parallel_w_gamma, workspace->parallel_w_inv,
            workspace->qcqp_diag_bq
                ? workspace->qc_diag_bq_w_gamma.values
                : NULL,
            workspace->qcqp_diag_bq
                ? workspace->qc_diag_bq_w_inv.values
                : NULL,
            anchor, reflection, !workspace->empty_A, workspace->to_check,
            workspace->n);
    }
    status = map_cuda_status(cudaGetLastError());
    if (status != HPRQP_STATUS_OK) return status;

    if (overlap_cone) {
        cudaStream_t cone_stream =
            (cudaStream_t)workspace->qc_cone_stream;
        cudaEvent_t x_ready =
            (cudaEvent_t)workspace->qc_x_ready_event;
        cudaEvent_t cone_done =
            (cudaEvent_t)workspace->qc_cone_done_event;
        status = map_cuda_status(cudaEventRecord(x_ready, (cudaStream_t)0));
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaStreamWaitEvent(
            cone_stream, x_ready, 0U));
        if (status != HPRQP_STATUS_OK) return status;
        qc_cone_update_kernel<<<update_blocks_p, update_threads_p,
                                0U, cone_stream>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            problem->qc_constant.values, problem->qc_offset.values,
            workspace->x_hat.values,
            workspace->qc_kappa.values, workspace->qc_theta.values,
            workspace->last_qc_dual.values, workspace->qc_dual.values,
            workspace->qc_dual_bar.values,
            workspace->qc_dual_delta.values,
            workspace->qc_primal_violation.values,
            workspace->qcqp_diag_bq
                ? workspace->sigma
                : workspace->sigma * workspace->d_s,
            workspace->qcqp_diag_bq
                ? workspace->qc_diag_bq_cone_metric.values
                : NULL,
            anchor, reflection,
            workspace->to_check, workspace->p);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
        status = map_cuda_status(cudaEventRecord(cone_done, cone_stream));
        if (status != HPRQP_STATUS_OK) return status;
    }

    if (!workspace->empty_A) {
        if (workspace->spmv_mode_a == HPRQP_SPMV_MODE_CUSTOMIZED &&
            workspace->custom_spmv_a_kernel == HPRQP_GPU_CUSTOM_SPMV_ROW) {
            qc_y_update_fused_a_kernel<<<update_blocks_m,
                                          update_threads_m>>>(
                problem->A.row_ptr, problem->A.col_ind, problem->A.values,
                workspace->x_hat.values, problem->AL.values,
                problem->AU.values, workspace->last_y.values,
                workspace->Ax.values, workspace->y.values,
                workspace->y_bar.values, workspace->dy.values,
                workspace->s.values, workspace->parallel_y_rho,
                workspace->parallel_y_inv, anchor, reflection,
                workspace->to_check, workspace->m);
        } else {
            status = use_spmv_graphs
                ? gpu_spmv_apply_a_x_hat_graph(workspace, problem)
                : gpu_spmv_apply_a_x_hat(workspace, problem);
            if (status != HPRQP_STATUS_OK) return status;
            qc_y_update_kernel<<<update_blocks_m,
                                 update_threads_m>>>(
                workspace->Ax.values, problem->AL.values,
                problem->AU.values, workspace->last_y.values,
                workspace->y.values, workspace->y_bar.values,
                workspace->dy.values, workspace->s.values,
                workspace->parallel_y_rho, workspace->parallel_y_inv,
                anchor, reflection, workspace->to_check, workspace->m);
        }
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
    }

    if (overlap_cone) {
        status = map_cuda_status(cudaStreamWaitEvent(
            (cudaStream_t)0,
            (cudaEvent_t)workspace->qc_cone_done_event, 0U));
        if (status != HPRQP_STATUS_OK) return status;
    } else {
        qc_cone_update_kernel<<<update_blocks_p, update_threads_p>>>(
            problem->qc_linear.row_ptr, problem->qc_linear.col_ind,
            problem->qc_linear.values, problem->qc_factor.row_ptr,
            problem->qc_factor.col_ind, problem->qc_factor.values,
            problem->qc_constant.values, problem->qc_offset.values,
            workspace->x_hat.values,
            workspace->qc_kappa.values, workspace->qc_theta.values,
            workspace->last_qc_dual.values, workspace->qc_dual.values,
            workspace->qc_dual_bar.values,
            workspace->qc_dual_delta.values,
            workspace->qc_primal_violation.values,
            workspace->qcqp_diag_bq
                ? workspace->sigma
                : workspace->sigma * workspace->d_s,
            workspace->qcqp_diag_bq
                ? workspace->qc_diag_bq_cone_metric.values
                : NULL,
            anchor, reflection,
            workspace->to_check, workspace->p);
        status = map_cuda_status(cudaGetLastError());
        if (status != HPRQP_STATUS_OK) return status;
    }

    if (workspace->to_check) {
        status = problem->Q.nnz > 0U
                     ? gpu_spmv_apply_q_w_bar(workspace, problem)
                     : map_cuda_status(cudaMemset(
                           workspace->Qw_bar.values, 0,
                           workspace->n * sizeof(double)));
        if (status != HPRQP_STATUS_OK) return status;
        if (!workspace->empty_A) {
            status = gpu_spmv_apply_at_y_bar(workspace, problem);
            if (status != HPRQP_STATUS_OK) return status;
        } else {
            status = map_cuda_status(cudaMemset(
                workspace->ATy_bar.values, 0,
                workspace->n * sizeof(double)));
            if (status != HPRQP_STATUS_OK) return status;
        }
        status = gpu_qcqp_apply_bt(workspace, problem,
                                   workspace->qc_dual_bar.values,
                                   workspace->BTs_bar.values);
        if (status != HPRQP_STATUS_OK) return status;
        qc_check_differences_kernel<<<workspace->blocks_n,
                                      workspace->threads_n>>>(
            workspace->Qw.values, workspace->Qw_bar.values,
            workspace->ATy.values, workspace->ATy_bar.values,
            workspace->BTs.values, workspace->BTs_bar.values,
            workspace->dQw.values, workspace->ATdy.values,
            workspace->BTds.values, !workspace->empty_A, workspace->n);
        status = map_cuda_status(cudaGetLastError());
    }
    return status;
}
