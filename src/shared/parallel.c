#include "shared/parallel.h"

#include <float.h>
#include <math.h>

static double clamp_double(double value, double lower, double upper)
{
    return value < lower ? lower : (value > upper ? upper : value);
}

HPRQPStatus parallel_validate_params(const HPRQPParams *params)
{
    if (params == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (params->method != HPRQP_METHOD_SGS &&
        params->method != HPRQP_METHOD_PARALLEL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (params->method == HPRQP_METHOD_SGS) {
        return HPRQP_STATUS_OK;
    }
    if (params->parallel_majorizer_mode != HPRQP_PARALLEL_MAJORIZER_WORST_CASE &&
        params->parallel_majorizer_mode != HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER &&
        params->parallel_majorizer_mode != HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT &&
        params->parallel_majorizer_mode != HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (params->qc_majorizer_mode != HPRQP_QC_MAJORIZER_AUTO &&
        params->qc_majorizer_mode != HPRQP_QC_MAJORIZER_PAIRWISE &&
        params->qc_majorizer_mode != HPRQP_QC_MAJORIZER_JOINT_F &&
        params->qc_majorizer_mode != HPRQP_QC_MAJORIZER_SCALAR_B &&
        params->qc_majorizer_mode != HPRQP_QC_MAJORIZER_DIAG_BQ) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (params->tau_mode < HPRQP_TAU_MODE_UNIT ||
        params->tau_mode > HPRQP_TAU_MODE_ADAPTIVE_RESTART) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if ((params->tau_mode == HPRQP_TAU_MODE_FIXED &&
         !(isfinite(params->tau) && params->tau > 0.0)) ||
        !(isfinite(params->tau_min) && params->tau_min > 0.0) ||
        !(isfinite(params->tau_max) && params->tau_max >= params->tau_min) ||
        !(isfinite(params->eig_factor) && params->eig_factor > 1.0) ||
        !(isfinite(params->metric_negative_tolerance) &&
          params->metric_negative_tolerance > 0.0) ||
        !(isfinite(params->parallel_adaptation_energy_floor) &&
          params->parallel_adaptation_energy_floor > 0.0)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    return HPRQP_STATUS_OK;
}

double parallel_select_tau(const HPRQPParams *params,
                           double lambda_max_A,
                           double lambda_max_Q)
{
    double tau_balance;

    if (params == NULL) {
        return NAN;
    }
    if (params->tau_mode == HPRQP_TAU_MODE_UNIT) {
        return 1.0;
    }
    if (params->tau_mode == HPRQP_TAU_MODE_FIXED) {
        return params->tau;
    }

    if (lambda_max_A > 0.0) {
        tau_balance = (lambda_max_Q * lambda_max_Q) / lambda_max_A;
    } else if (lambda_max_Q > 0.0) {
        tau_balance = INFINITY;
    } else {
        tau_balance = 1.0;
    }
    return clamp_double(tau_balance, params->tau_min, params->tau_max);
}

double parallel_normalized_coupling(double lambda_max_AQA,
                                    double lambda_max_A,
                                    double lambda_max_Q)
{
    double denominator = lambda_max_A * lambda_max_Q;
    double ratio;

    if (!(denominator > 0.0)) {
        return 0.0;
    }
    ratio = lambda_max_AQA / denominator;
    return sqrt(clamp_double(ratio, 0.0, 1.0));
}

double parallel_second_moment_delta(double lambda_max_Q,
                                    double lambda_max_AQA,
                                    double lambda_max_AQ2A)
{
    double denominator;
    double ratio;

    if (!(isfinite(lambda_max_Q) && lambda_max_Q >= 0.0 &&
          isfinite(lambda_max_AQA) && lambda_max_AQA >= 0.0 &&
          isfinite(lambda_max_AQ2A) && lambda_max_AQ2A >= 0.0)) {
        return NAN;
    }
    denominator = lambda_max_Q * lambda_max_AQA;
    if (!(denominator > 0.0)) return 0.0;
    ratio = clamp_double(lambda_max_AQ2A / denominator, 0.0, 1.0);
    return 1.0 - ratio;
}

HPRQPParallelMajorizerMode parallel_select_moment_majorizer(
    double lambda_max_Q,
    double lambda_max_AQA,
    double lambda_max_AQ2A,
    double *delta_2)
{
    double delta = parallel_second_moment_delta(lambda_max_Q,
                                                lambda_max_AQA,
                                                lambda_max_AQ2A);
    if (delta_2 != NULL) *delta_2 = delta;
    if (!isfinite(delta)) return HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT;
    return delta <= HPRQP_SECOND_MOMENT_DELTA_TOLERANCE
               ? HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER
               : HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT;
}

static double two_moment_derivative(double r,
                                    double lambda_max_Q,
                                    double lambda_max_AQA,
                                    double lambda_max_AQ2A,
                                    double y_displacement_sq,
                                    double w_q_displacement)
{
    double delta = r - lambda_max_Q;
    double r_sq;
    double delta_sq;
    double curvature = 0.0;

    if (!(isfinite(r) && r > lambda_max_Q &&
          isfinite(lambda_max_AQA) && lambda_max_AQA >= 0.0 &&
          isfinite(lambda_max_AQ2A) && lambda_max_AQ2A >= 0.0 &&
          isfinite(y_displacement_sq) && y_displacement_sq >= 0.0 &&
          isfinite(w_q_displacement) && w_q_displacement >= 0.0)) {
        return NAN;
    }
    r_sq = r * r;
    delta_sq = delta * delta;
    if (lambda_max_AQA > 0.0) curvature += lambda_max_AQA / r_sq;
    if (lambda_max_AQ2A > 0.0) {
        curvature += lambda_max_AQ2A * (2.0 * r - lambda_max_Q) /
                     (r_sq * delta_sq);
    }
    if (!isfinite(curvature)) return -INFINITY;
    return w_q_displacement - y_displacement_sq * curvature;
}

double parallel_select_two_moment_tau(
    const HPRQPParams *params,
    double lambda_max_A,
    double lambda_max_Q,
    double lambda_max_AQA,
    double lambda_max_AQ2A,
    double y_displacement_sq,
    double w_q_displacement)
{
    const double tau_min = params != NULL ? params->tau_min : NAN;
    const double tau_max = params != NULL ? params->tau_max : NAN;
    double fallback;
    double zeta_AQ;
    double displacement_scale;
    double y_scaled;
    double w_scaled;
    double r_low;
    double r_high = NAN;
    double r_high_delta;
    double derivative_low;
    double derivative_high = NAN;
    double r_mid = NAN;
    double tau_raw;
    size_t iteration;

    if (params == NULL) return NAN;
    fallback = parallel_select_tau(params, lambda_max_A, lambda_max_Q);
    if (params->tau_mode != HPRQP_TAU_MODE_ADAPTIVE_RESTART) return fallback;
    fallback = clamp_double(fallback, tau_min, tau_max);
    if (!(isfinite(lambda_max_A) && lambda_max_A >= 0.0 &&
          isfinite(lambda_max_Q) && lambda_max_Q >= 0.0 &&
          isfinite(lambda_max_AQA) && lambda_max_AQA >= 0.0 &&
          isfinite(lambda_max_AQ2A) && lambda_max_AQ2A >= 0.0 &&
          isfinite(y_displacement_sq) && y_displacement_sq >= 0.0 &&
          isfinite(w_q_displacement))) return fallback;

    zeta_AQ = parallel_normalized_coupling(lambda_max_AQA,
                                           lambda_max_A,
                                           lambda_max_Q);
    if (!(zeta_AQ > 0.0 && lambda_max_Q > 0.0)) return fallback;
    if (!(y_displacement_sq > 0.0 && w_q_displacement > 0.0)) {
        return clamp_double(1.0, tau_min, tau_max);
    }
    displacement_scale = fmax(y_displacement_sq, w_q_displacement);
    y_scaled = y_displacement_sq / displacement_scale;
    w_scaled = w_q_displacement / displacement_scale;
    r_low = nextafter(lambda_max_Q, INFINITY);
    derivative_low = two_moment_derivative(r_low, lambda_max_Q,
                                           lambda_max_AQA,
                                           lambda_max_AQ2A,
                                           y_scaled, w_scaled);
    if (isnan(derivative_low)) return fallback;
    if (derivative_low >= 0.0) return tau_max;

    r_high_delta = fmax(lambda_max_Q, 1.0);
    for (iteration = 0; iteration < 1024; ++iteration) {
        r_high = lambda_max_Q + r_high_delta;
        derivative_high = two_moment_derivative(r_high, lambda_max_Q,
                                                lambda_max_AQA,
                                                lambda_max_AQ2A,
                                                y_scaled, w_scaled);
        if (isnan(derivative_high)) return fallback;
        if (derivative_high >= 0.0) break;
        if (r_high_delta > 0.25 * DBL_MAX) {
            derivative_high = -INFINITY;
            break;
        }
        r_high_delta *= 2.0;
    }
    if (!(derivative_high >= 0.0)) return tau_min;
    for (iteration = 0; iteration < 80; ++iteration) {
        double derivative_mid;
        r_mid = 0.5 * (r_low + r_high);
        derivative_mid = two_moment_derivative(r_mid, lambda_max_Q,
                                               lambda_max_AQA,
                                               lambda_max_AQ2A,
                                               y_scaled, w_scaled);
        if (derivative_mid < 0.0) r_low = r_mid;
        else r_high = r_mid;
    }
    r_mid = 0.5 * (r_low + r_high);
    tau_raw = zeta_AQ * lambda_max_Q / (r_mid - lambda_max_Q);
    if (isinf(tau_raw) && tau_raw > 0.0) return tau_max;
    if (tau_raw == 0.0) return tau_min;
    return isfinite(tau_raw) && tau_raw > 0.0
               ? clamp_double(tau_raw, tau_min, tau_max)
               : fallback;
}

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
                                          double *w_inv)
{
    double coupling;

    if (d_y == NULL || d_w == NULL || y_rho == NULL || y_inv == NULL ||
        w_gamma == NULL || w_inv == NULL ||
        !(isfinite(sigma) && sigma > 0.0) ||
        !(isfinite(tau) && tau > 0.0) ||
        !(isfinite(lambda_max_A) && lambda_max_A >= 0.0) ||
        !(isfinite(lambda_max_Q) && lambda_max_Q >= 0.0) ||
        majorizer_mode == HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR ||
        (majorizer_mode == HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT &&
         (!(isfinite(lambda_max_AQA) && lambda_max_AQA >= 0.0) ||
          !(isfinite(lambda_max_AQ2A) && lambda_max_AQ2A >= 0.0)))) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    coupling = majorizer_mode == HPRQP_PARALLEL_MAJORIZER_WORST_CASE
                   ? 1.0 : zeta_AQ;
    if (!(isfinite(coupling) && coupling >= 0.0 && coupling <= 1.0)) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    if (majorizer_mode == HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT &&
        coupling > 0.0 && lambda_max_Q > 0.0) {
        double r = lambda_max_Q * (1.0 + coupling / tau);
        double delta = r - lambda_max_Q;
        double compatible_AQ2A = fmin(lambda_max_AQ2A,
                                      lambda_max_Q * lambda_max_AQA);
        double ell;
        if (!(r > lambda_max_Q && delta > 0.0)) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        ell = lambda_max_A + lambda_max_AQA / r +
              compatible_AQ2A / (r * delta);
        if (!(isfinite(ell) && ell > 0.0 && lambda_max_A > 0.0)) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        *d_y = ell / lambda_max_A;
        *d_w = r / lambda_max_Q;
    } else {
        *d_y = 1.0 + coupling * tau;
        *d_w = 1.0 + coupling / tau;
    }
    if (m > 0) {
        if (!(lambda_max_A > 0.0)) {
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }
        *y_rho = sigma * (*d_y) * lambda_max_A;
        *y_inv = 1.0 / (*y_rho);
    } else {
        *y_rho = 0.0;
        *y_inv = 0.0;
    }
    *w_gamma = sigma * (*d_w) * lambda_max_Q;
    *w_inv = 1.0 / (1.0 + *w_gamma);
    return HPRQP_STATUS_OK;
}
