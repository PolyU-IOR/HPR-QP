#include "hprqp.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifndef HPRQP_SOURCE_DIR
#define HPRQP_SOURCE_DIR "."
#endif

static double wall_seconds(void)
{
    struct timespec value;

    if (timespec_get(&value, TIME_UTC) != TIME_UTC) {
        return 0.0;
    }
    return (double)value.tv_sec + 1e-9 * (double)value.tv_nsec;
}

static const char *status_name(HPRQPStatus status)
{
    switch (status) {
    case HPRQP_STATUS_OK:
        return "OK";
    case HPRQP_STATUS_OPTIMAL:
        return "OPTIMAL";
    case HPRQP_STATUS_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case HPRQP_STATUS_ALLOCATION_FAILED:
        return "ALLOCATION_FAILED";
    case HPRQP_STATUS_NOT_IMPLEMENTED:
        return "NOT_IMPLEMENTED";
    case HPRQP_STATUS_CONTINUE:
        return "CONTINUE";
    case HPRQP_STATUS_MAX_ITER:
        return "MAX_ITER";
    case HPRQP_STATUS_TIME_LIMIT:
        return "TIME_LIMIT";
    default:
        return "UNKNOWN";
    }
}

static int parse_size_arg(const char *value, size_t *out)
{
    char *endptr;
    unsigned long long parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtoull(value, &endptr, 10);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = (size_t)parsed;
    return 1;
}

static int parse_int_arg(const char *value, int *out)
{
    char *endptr;
    long parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtol(value, &endptr, 10);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = (int)parsed;
    return 1;
}

static int parse_double_arg(const char *value, double *out)
{
    char *endptr;
    double parsed;

    if (value == NULL || out == NULL) {
        return 0;
    }

    parsed = strtod(value, &endptr);
    if (*value == '\0' || *endptr != '\0') {
        return 0;
    }

    *out = parsed;
    return 1;
}

static int parse_spmv_mode_arg(const char *value, HPRQPSpmvMode *out)
{
    if (value == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(value, "auto") == 0) {
        *out = HPRQP_SPMV_MODE_AUTO;
        return 1;
    }
    if (strcmp(value, "cusparse") == 0) {
        *out = HPRQP_SPMV_MODE_CUSPARSE;
        return 1;
    }
    if (strcmp(value, "customized") == 0) {
        *out = HPRQP_SPMV_MODE_CUSTOMIZED;
        return 1;
    }
    return 0;
}

static int parse_method_arg(const char *value, HPRQPMethod *out)
{
    if (value == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(value, "sgs") == 0) {
        *out = HPRQP_METHOD_SGS;
        return 1;
    }
    if (strcmp(value, "parallel") == 0) {
        *out = HPRQP_METHOD_PARALLEL;
        return 1;
    }
    return 0;
}

static int parse_parallel_majorizer_arg(const char *value,
                                        HPRQPParallelMajorizerMode *out)
{
    if (value == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(value, "worst_case") == 0 ||
        strcmp(value, "worst-case") == 0) {
        *out = HPRQP_PARALLEL_MAJORIZER_WORST_CASE;
        return 1;
    }
    if (strcmp(value, "coupling_power") == 0 ||
        strcmp(value, "coupling-power") == 0) {
        *out = HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER;
        return 1;
    }
    if (strcmp(value, "two_moment") == 0 ||
        strcmp(value, "two-moment") == 0) {
        *out = HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT;
        return 1;
    }
    if (strcmp(value, "moment_selector") == 0 ||
        strcmp(value, "moment-selector") == 0) {
        *out = HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR;
        return 1;
    }
    return 0;
}

static int parse_qc_majorizer_arg(const char *value,
                                  HPRQPQCMajorizerMode *out)
{
    if (value == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(value, "auto") == 0) {
        *out = HPRQP_QC_MAJORIZER_AUTO;
        return 1;
    }
    if (strcmp(value, "pairwise") == 0) {
        *out = HPRQP_QC_MAJORIZER_PAIRWISE;
        return 1;
    }
    if (strcmp(value, "joint_f") == 0 ||
        strcmp(value, "joint-f") == 0) {
        *out = HPRQP_QC_MAJORIZER_JOINT_F;
        return 1;
    }
    if (strcmp(value, "scalar_b") == 0 ||
        strcmp(value, "scalar-b") == 0) {
        *out = HPRQP_QC_MAJORIZER_SCALAR_B;
        return 1;
    }
    if (strcmp(value, "diag_bq") == 0 ||
        strcmp(value, "diag-bq") == 0) {
        *out = HPRQP_QC_MAJORIZER_DIAG_BQ;
        return 1;
    }
    return 0;
}

static int parse_tau_mode_arg(const char *value, HPRQPTauMode *out)
{
    if (value == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(value, "unit") == 0) {
        *out = HPRQP_TAU_MODE_UNIT;
        return 1;
    }
    if (strcmp(value, "fixed") == 0) {
        *out = HPRQP_TAU_MODE_FIXED;
        return 1;
    }
    if (strcmp(value, "balanced") == 0) {
        *out = HPRQP_TAU_MODE_BALANCED;
        return 1;
    }
    if (strcmp(value, "adaptive_restart") == 0 ||
        strcmp(value, "adaptive-restart") == 0) {
        *out = HPRQP_TAU_MODE_ADAPTIVE_RESTART;
        return 1;
    }
    return 0;
}

static void print_usage(const char *argv0)
{
    printf("usage: %s [mps_path] [--method sgs|parallel] [--parallel-majorizer worst_case|coupling_power|two_moment|moment_selector] [--qc-majorizer auto|pairwise|joint_f|scalar_b|diag_bq] [--tau-mode unit|fixed|balanced|adaptive_restart] [--tau X] [--tau-min X] [--tau-max X] [--sigma-fixed 0|1] [--initial-sigma X] [--use-socp-restart-policy 0|1] [--use-socp-sigma-policy 0|1] [--max-iter N] [--stoptol X] [--check-iter N] [--print-frequency N] [--time-limit X] [--verbose 0|1] [--use-gpu 0|1] [--device-number N] [--spmv-mode-q auto|cusparse|customized] [--spmv-mode-a auto|cusparse|customized] [--use-scaling 0|1] [--use-cr-scaling 0|1] [--cr-iterations N] [--use-bc-scaling 0|1]\n", argv0);
}

int main(int argc, char **argv)
{
    HPRQPModel model;
    HPRQPParams params;
    HPRQPResults results;
    HPRQPStatus status;
    const char *mps_path = HPRQP_SOURCE_DIR "/model.mps";
    const int profile_setup = getenv("HPRQP_PROFILE_SETUP") != NULL;
    double process_start = profile_setup ? wall_seconds() : 0.0;
    double mps_done;
    double solve_done;
    double print_done;
    int argi = 1;

    model_init(&model);
    results_init(&results);
    params_set_defaults(&params);
    while (argi < argc) {
        if (strcmp(argv[argi], "--method") == 0 && argi + 1 < argc) {
            if (!parse_method_arg(argv[argi + 1], &params.method)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--parallel-majorizer") == 0 && argi + 1 < argc) {
            if (!parse_parallel_majorizer_arg(argv[argi + 1], &params.parallel_majorizer_mode)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--qc-majorizer") == 0 && argi + 1 < argc) {
            if (!parse_qc_majorizer_arg(argv[argi + 1], &params.qc_majorizer_mode)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--tau-mode") == 0 && argi + 1 < argc) {
            if (!parse_tau_mode_arg(argv[argi + 1], &params.tau_mode)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--tau") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &params.tau)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--tau-min") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &params.tau_min)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--tau-max") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &params.tau_max)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--sigma-fixed") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.sigma_fixed)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--initial-sigma") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &params.initial_sigma)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-socp-restart-policy") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_socp_restart_policy)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-socp-sigma-policy") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_socp_sigma_policy)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--max-iter") == 0 && argi + 1 < argc) {
            if (!parse_size_arg(argv[argi + 1], &params.max_iter)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--stoptol") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &params.stoptol)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--check-iter") == 0 && argi + 1 < argc) {
            if (!parse_size_arg(argv[argi + 1], &params.check_iter)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--print-frequency") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.print_frequency)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--time-limit") == 0 && argi + 1 < argc) {
            if (!parse_double_arg(argv[argi + 1], &params.time_limit)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--verbose") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.verbose)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-gpu") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_gpu)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--device-number") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.device_number)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--spmv-mode-q") == 0 && argi + 1 < argc) {
            if (!parse_spmv_mode_arg(argv[argi + 1], &params.spmv_mode_q)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--spmv-mode-a") == 0 && argi + 1 < argc) {
            if (!parse_spmv_mode_arg(argv[argi + 1], &params.spmv_mode_a)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-scaling") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_scaling)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-cr-scaling") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_cr_scaling)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--cr-iterations") == 0 && argi + 1 < argc) {
            if (!parse_size_arg(argv[argi + 1], &params.cr_iterations)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (strcmp(argv[argi], "--use-bc-scaling") == 0 && argi + 1 < argc) {
            if (!parse_int_arg(argv[argi + 1], &params.use_bc_scaling)) {
                print_usage(argv[0]);
                return 1;
            }
            argi += 2;
        } else if (argv[argi][0] != '-') {
            mps_path = argv[argi];
            argi += 1;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    status = build_from_mps(mps_path, &model);
    if (profile_setup) {
        mps_done = wall_seconds();
    }
    if (status != HPRQP_STATUS_OK) {
        fprintf(stderr, "failed to load MPS file '%s' with status %s\n", mps_path, status_name(status));
        return 1;
    }

    status = solve(&model, &params, &results);
    if (profile_setup) {
        solve_done = wall_seconds();
    }
    printf("model: %s\n", mps_path);
    printf("status: %s\n", status_name(status));
    printf("iter: %zu\n", results.iter);
    printf("time: %.6f\n", results.time);
    printf("objective: %.12g\n", results.primal_obj);
    printf("residual: %.12g\n", results.residuals);
    printf("gap: %.12g\n", results.gap);
    if (results.method == HPRQP_METHOD_PARALLEL) {
        printf("lambda_max_A: %.12g\n", results.lambda_max_A);
        printf("lambda_max_Q: %.12g\n", results.lambda_max_Q);
        printf("lambda_max_AQA: %.12g\n", results.lambda_max_AQA);
        printf("lambda_max_AQ2A: %.12g\n", results.lambda_max_AQ2A);
        if (results.qc_cone_dim > 0U) {
            if (params.qc_majorizer_mode == HPRQP_QC_MAJORIZER_JOINT_F) {
                printf("zeta_FQ: %.12g\n", results.zeta_AQ);
                printf("tau_FQ: %.12g\n", results.tau_AQ);
            } else {
                printf("zeta_AB: %.12g\n", results.zeta_AB);
                printf("zeta_AQ: %.12g\n", results.zeta_AQ);
                printf("zeta_BQ: %.12g\n", results.zeta_BQ);
                printf("tau_AB: %.12g\n", results.tau_AB);
                printf("tau_AQ: %.12g\n", results.tau_AQ);
                printf("tau_BQ: %.12g\n", results.tau_BQ);
            }
            printf("d_s: %.12g\n", results.d_s);
        } else {
            printf("zeta_AQ: %.12g\n", results.zeta_AQ);
            printf("tau: %.12g\n", results.tau);
        }
        printf("d_y: %.12g\n", results.d_y);
        printf("d_w: %.12g\n", results.d_w);
    }
    printf("iter_4: %zu\n", results.iter_4);
    printf("time_4: %.6f\n", results.time_4);
    printf("iter_6: %zu\n", results.iter_6);
    printf("time_6: %.6f\n", results.time_6);
    if (profile_setup) {
        print_done = wall_seconds();
    }

    // if (results.x != NULL) {
    //     printf("x:");
    //     for (i = 0; i < model.Q.ncol; ++i) {
    //         printf(" %.12g", results.x[i]);
    //     }
    //     printf("\n");
    // }

    // if (results.y != NULL && model.A.nrow > 0) {
    //     printf("y:");
    //     for (i = 0; i < model.A.nrow; ++i) {
    //         printf(" %.12g", results.y[i]);
    //     }
    //     printf("\n");
    // }

    results_free(&results);
    model_free(&model);
    if (profile_setup) {
        double process_done = wall_seconds();
        fprintf(stderr,
                "HPRQP_PROFILE process: mps_and_build=%.6f s, "
                "solve_call=%.6f s, result_print=%.6f s, "
                "cleanup=%.6f s, total=%.6f s\n",
                mps_done - process_start,
                solve_done - mps_done,
                print_done - solve_done,
                process_done - print_done,
                process_done - process_start);
    }
    return status == HPRQP_STATUS_OPTIMAL || status == HPRQP_STATUS_MAX_ITER || status == HPRQP_STATUS_TIME_LIMIT ? 0 : 1;
}
