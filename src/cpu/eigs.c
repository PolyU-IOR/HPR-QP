#include "cpu/eigs.h"

#include "shared/sparse.h"
#include "shared/parallel.h"
#include "shared/vector.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const double HPRQP_EIGENVALUE_SCALE = 1.05;

static void init_probe(double *x, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        x[i] = 1.0 + ((double)i + 1.0) / ((double)n + 1.0);
    }
}

static HPRQPStatus power_iteration_a(const HPRQPWorkspace *workspace,
                                     size_t max_iterations,
                                     double tolerance,
                                     double *lambda_max)
{
    double *z;
    double *q;
    double *at_q;
    size_t iteration;

    if (workspace->m == 0) {
        *lambda_max = 0.0;
        return HPRQP_STATUS_OK;
    }

    z = (double *)malloc(workspace->m * sizeof(double));
    q = (double *)malloc(workspace->m * sizeof(double));
    at_q = (double *)malloc(workspace->n * sizeof(double));
    if (z == NULL || q == NULL || at_q == NULL) {
        free(z);
        free(q);
        free(at_q);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    init_probe(z, workspace->m);
    *lambda_max = 1.0;

    for (iteration = 0; iteration < max_iterations; ++iteration) {
        size_t i;
        double q_norm = vec_norm2(z, workspace->m);
        double z_norm;
        double error;

        if (q_norm <= 1e-16) {
            *lambda_max = 0.0;
            break;
        }

        for (i = 0; i < workspace->m; ++i) {
            q[i] = z[i] / q_norm;
        }

        if (matrix_csr_matvec(workspace->AT, q, at_q) != HPRQP_STATUS_OK ||
            matrix_csr_matvec(workspace->A, at_q, z) != HPRQP_STATUS_OK) {
            free(z);
            free(q);
            free(at_q);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }

        *lambda_max = vec_dot(q, z, workspace->m);
        z_norm = vec_norm2(z, workspace->m);
        for (i = 0; i < workspace->m; ++i) {
            q[i] = z[i] - (*lambda_max) * q[i];
        }
        error = vec_norm2(q, workspace->m) / (z_norm + fabs(*lambda_max) + 1e-16);
        if (error < tolerance) {
            break;
        }
    }

    free(z);
    free(q);
    free(at_q);
    if (*lambda_max < 0.0) {
        *lambda_max = 0.0;
    }
    return HPRQP_STATUS_OK;
}

static HPRQPStatus power_iteration_q(const HPRQPWorkspace *workspace,
                                     size_t max_iterations,
                                     double tolerance,
                                     double *lambda_max)
{
    double *z;
    double *q;
    size_t iteration;

    if (workspace->n == 0) {
        *lambda_max = 0.0;
        return HPRQP_STATUS_OK;
    }

    z = (double *)malloc(workspace->n * sizeof(double));
    q = (double *)malloc(workspace->n * sizeof(double));
    if (z == NULL || q == NULL) {
        free(z);
        free(q);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    init_probe(z, workspace->n);
    *lambda_max = 1.0;

    for (iteration = 0; iteration < max_iterations; ++iteration) {
        size_t i;
        double q_norm = vec_norm2(z, workspace->n);
        double z_norm;
        double error;

        if (q_norm <= 1e-16) {
            *lambda_max = 0.0;
            break;
        }

        for (i = 0; i < workspace->n; ++i) {
            q[i] = z[i] / q_norm;
        }

        if (matrix_csr_matvec(workspace->Q, q, z) != HPRQP_STATUS_OK) {
            free(z);
            free(q);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }

        *lambda_max = vec_dot(q, z, workspace->n);
        z_norm = vec_norm2(z, workspace->n);
        for (i = 0; i < workspace->n; ++i) {
            q[i] = z[i] - (*lambda_max) * q[i];
        }
        error = vec_norm2(q, workspace->n) / (z_norm + fabs(*lambda_max) + 1e-16);
        if (error < tolerance) {
            break;
        }
    }

    free(z);
    free(q);
    if (*lambda_max < 0.0) {
        *lambda_max = 0.0;
    }
    return HPRQP_STATUS_OK;
}

static HPRQPStatus power_iteration_aqka(const HPRQPWorkspace *workspace,
                                        size_t max_iterations,
                                        double tolerance,
                                        int q_power,
                                        double *lambda_max)
{
    double *z;
    double *q;
    double *at_q;
    double *q_at_q;
    size_t iteration;

    if (q_power < 1 || q_power > 2) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    if (workspace->m == 0 || workspace->n == 0 || workspace->Q->nnz == 0) {
        *lambda_max = 0.0;
        return HPRQP_STATUS_OK;
    }

    z = (double *)malloc(workspace->m * sizeof(double));
    q = (double *)malloc(workspace->m * sizeof(double));
    at_q = (double *)malloc(workspace->n * sizeof(double));
    q_at_q = (double *)malloc(workspace->n * sizeof(double));
    if (z == NULL || q == NULL || at_q == NULL || q_at_q == NULL) {
        free(z);
        free(q);
        free(at_q);
        free(q_at_q);
        return HPRQP_STATUS_ALLOCATION_FAILED;
    }

    init_probe(z, workspace->m);
    *lambda_max = 1.0;
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        size_t i;
        double q_norm = vec_norm2(z, workspace->m);
        double z_norm;
        double error;

        if (q_norm <= 1e-16) {
            *lambda_max = 0.0;
            break;
        }
        for (i = 0; i < workspace->m; ++i) {
            q[i] = z[i] / q_norm;
        }
        if (matrix_csr_matvec(workspace->AT, q, at_q) != HPRQP_STATUS_OK ||
            matrix_csr_matvec(workspace->Q, at_q, q_at_q) != HPRQP_STATUS_OK ||
            (q_power == 2 &&
             matrix_csr_matvec(workspace->Q, q_at_q, at_q) != HPRQP_STATUS_OK) ||
            matrix_csr_matvec(workspace->A,
                              q_power == 2 ? at_q : q_at_q,
                              z) != HPRQP_STATUS_OK) {
            free(z);
            free(q);
            free(at_q);
            free(q_at_q);
            return HPRQP_STATUS_INVALID_ARGUMENT;
        }

        *lambda_max = vec_dot(q, z, workspace->m);
        z_norm = vec_norm2(z, workspace->m);
        for (i = 0; i < workspace->m; ++i) {
            q[i] = z[i] - (*lambda_max) * q[i];
        }
        error = vec_norm2(q, workspace->m) /
                (z_norm + fabs(*lambda_max) + 1e-16);
        if (error < tolerance) {
            break;
        }
    }

    free(z);
    free(q);
    free(at_q);
    free(q_at_q);
    if (*lambda_max < 0.0) {
        *lambda_max = 0.0;
    }
    return HPRQP_STATUS_OK;
}

HPRQPStatus estimate_eigenvalues(HPRQPWorkspace *workspace,
                                 const HPRQPParams *params)
{
    HPRQPStatus status;
    double eig_factor = params != NULL ? params->eig_factor : HPRQP_EIGENVALUE_SCALE;
    int moment_selector_requested;

    if (workspace == NULL || workspace->Q == NULL || workspace->A == NULL || workspace->AT == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }
    moment_selector_requested =
        workspace->parallel_majorizer_mode ==
        HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR;

    status = power_iteration_a(workspace, 5000, 1e-4, &workspace->lambda_max_A);
    if (status != HPRQP_STATUS_OK) {
        return status;
    }
    workspace->lambda_max_A *= eig_factor;

    if (workspace->Q_is_diag) {
        size_t i;
        workspace->lambda_max_Q = 0.0;
        for (i = 0; i < workspace->n; ++i) {
            if (workspace->diag_Q[i] > workspace->lambda_max_Q) {
                workspace->lambda_max_Q = workspace->diag_Q[i];
            }
        }
    } else {
        status = power_iteration_q(workspace, 5000, 1e-4, &workspace->lambda_max_Q);
        if (status != HPRQP_STATUS_OK) {
            return status;
        }
        workspace->lambda_max_Q *= eig_factor;
    }

    if (workspace->method == HPRQP_METHOD_PARALLEL) {
        if (workspace->parallel_majorizer_mode ==
                HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER ||
            workspace->parallel_majorizer_mode ==
                HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT ||
            moment_selector_requested) {
            if (workspace->m > 0 && workspace->lambda_max_A > 0.0 &&
                workspace->lambda_max_Q > 0.0) {
                status = power_iteration_aqka(workspace,
                                              5000,
                                              1e-4,
                                              1,
                                              &workspace->lambda_max_AQA);
                if (status != HPRQP_STATUS_OK) {
                    return status;
                }
                workspace->lambda_max_AQA *= eig_factor;
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
                    status = power_iteration_aqka(workspace,
                                                  5000,
                                                  1e-4,
                                                  2,
                                                  &workspace->lambda_max_AQ2A);
                    if (status != HPRQP_STATUS_OK) return status;
                    workspace->lambda_max_AQ2A *= eig_factor;
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
                if (params != NULL && params->verbose) {
                    fprintf(stderr,
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
                    fflush(stderr);
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
    return HPRQP_STATUS_OK;
}
