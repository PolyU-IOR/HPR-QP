#include "hprqp.h"

#include "cpu/scale.h"
#include "gpu/context.h"
#include "gpu/control.h"
#include "gpu/qcqp.h"
#include "gpu/scale.h"
#include "gpu/spmv.h"
#include "gpu/workspace.h"

#include <math.h>
#include <stdio.h>

#include <cuda_runtime_api.h>

static int nearly_equal(double actual, double expected, double tolerance)
{
    return fabs(actual - expected) <=
           tolerance * fmax(1.0, fmax(fabs(actual), fabs(expected)));
}

static int test_active_qcqp_solver(void)
{
    const char *path = HPRQP_SOURCE_DIR "/tests/data/active_qcqp.mps";
    HPRQPModel model;
    HPRQPParams params;
    HPRQPResults results;
    HPRQPStatus status;
    int ok = 0;

    model_init(&model);
    params_set_defaults(&params);
    results_init(&results);
    params.use_gpu = 1;
    params.device_number = 0;
    params.method = HPRQP_METHOD_PARALLEL;
    params.use_scaling = 1;
    params.use_cr_scaling = 1;
    params.cr_iterations = 10;
    params.check_iter = 10;
    params.print_frequency = 0;
    params.verbose = 0;

    status = build_from_mps(path, &model);
    if (status != HPRQP_STATUS_OK) goto cleanup;

    /* The structural selector must not consume an iteration while tuning. */
    params.stoptol = 0.0;
    params.max_iter = 0;
    status = solve_gpu(&model, &params, &results);
    if (status != HPRQP_STATUS_MAX_ITER ||
        !nearly_equal(results.primal_obj, 0.0, 1e-14) ||
        !nearly_equal(results.gap, 0.0, 1e-14)) {
        fprintf(stderr,
                "QCQP selector changed iteration-zero state: status=%d, "
                "pobj=%g, gap=%g\n",
                status, results.primal_obj, results.gap);
        goto cleanup;
    }
    results_free(&results);

    params.stoptol = 1e-8;
    params.max_iter = 10000;
    status = solve_gpu(&model, &params, &results);
    if (status != HPRQP_STATUS_OPTIMAL || results.x == NULL ||
        !nearly_equal(results.x[0], 1.0 / sqrt(2.0), 5e-7) ||
        !nearly_equal(results.primal_obj, 0.25 - sqrt(2.0), 5e-8) ||
        results.primal_residual > 1e-8 || results.qc_residual > 1e-8 ||
        results.dual_residual > 1e-8 || results.gap > 1e-8 ||
        results.qc_dual == NULL || results.qc_cone_dim != 3U) {
        fprintf(stderr,
                "active QCQP solve failed: status=%d, x=%g, obj=%g, "
                "rp=%g, rqc=%g, rd=%g, gap=%g\n",
                status, results.x != NULL ? results.x[0] : NAN,
                results.primal_obj, results.primal_residual,
                results.qc_residual, results.dual_residual, results.gap);
        goto cleanup;
    }
    ok = 1;

cleanup:
    results_free(&results);
    model_free(&model);
    return ok ? 0 : 1;
}

static int test_qc_affine_cone_scaling(void)
{
    const char *path = HPRQP_SOURCE_DIR "/tests/data/qc_scaling.mps";
    HPRQPModel model;
    HPRQPParams params;
    HPRQPScaling scaling;
    HPRQPGPUContext context;
    HPRQPGPUProblem problem;
    HPRQPGPUWorkspace workspace;
    HPRQPStatus status;
    double col_scale[2] = {0.0, 0.0};
    double qc_linear[2] = {0.0, 0.0};
    double qc_factor[2] = {0.0, 0.0};
    double qc_constant[2] = {0.0, 0.0};
    double qc_offset[2] = {0.0, 0.0};
    double qc_cr_shape[2] = {0.0, 0.0};
    int ok = 0;

    model_init(&model);
    params_set_defaults(&params);
    scaling_init(&scaling);
    gpu_context_init(&context);
    gpu_problem_init(&problem);
    gpu_workspace_init(&workspace);
    params.use_gpu = 1;
    params.use_scaling = 1;
    params.use_cr_scaling = 1;
    params.cr_iterations = 10;
    params.verbose = 0;

    status = build_from_mps(path, &model);
    if (status != HPRQP_STATUS_OK || model.qc.count != 2U ||
        model.qc.cone_ptr[1] - model.qc.cone_ptr[0] != 3U ||
        model.qc.cone_ptr[2] - model.qc.cone_ptr[1] != 3U) {
        goto cleanup;
    }
    status = gpu_context_open(&context, 0);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_problem_upload(&problem, &model);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_workspace_allocate(&workspace, &problem, &params);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_problem_scale(&problem, &workspace, &params, &scaling);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_to_host(&workspace.col_norm, col_scale, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_to_host(&problem.qc_constant,
                                     qc_constant, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_to_host(&problem.qc_offset, qc_offset, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_to_host(&workspace.qc_cr_shape,
                                     qc_cr_shape, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    if (cudaMemcpy(qc_linear, problem.qc_linear.values,
                   sizeof(qc_linear), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(qc_factor, problem.qc_factor.values,
                   sizeof(qc_factor), cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto cleanup;
    }

    if (!nearly_equal(scaling.norm_b_org, 0.0, 1e-14) ||
        !nearly_equal(workspace.scale_norm_b_org, 0.0, 1e-14) ||
        !(col_scale[0] > 1e2 && col_scale[1] > 1e2) ||
        !(qc_constant[0] > 0.0 && qc_constant[1] > 0.0) ||
        !isfinite(qc_constant[0]) || !isfinite(qc_constant[1]) ||
        !(qc_cr_shape[0] > 0.0 && qc_cr_shape[1] > 0.0) ||
        !isfinite(qc_cr_shape[0]) || !isfinite(qc_cr_shape[1]) ||
        (nearly_equal(qc_cr_shape[0], 1.0, 1e-12) &&
         nearly_equal(qc_cr_shape[1], 1.0, 1e-12)) ||
        nearly_equal(qc_constant[0], 1.0, 1e-12) ||
        nearly_equal(qc_constant[1], 1.0, 1e-12) ||
        !nearly_equal(qc_linear[0] * col_scale[0] /
                          (qc_constant[0] * qc_cr_shape[0]),
                      1e8, 2e-11) ||
        !nearly_equal(qc_linear[1] * col_scale[1] /
                          (qc_constant[1] * qc_cr_shape[1]),
                      1.0, 2e-11) ||
        !nearly_equal(qc_factor[0] * col_scale[0] /
                          (qc_constant[0] * sqrt(qc_cr_shape[0])),
                      sqrt(2.0), 2e-11) ||
        !nearly_equal(qc_factor[1] * col_scale[1] /
                          (qc_constant[1] * sqrt(qc_cr_shape[1])),
                      sqrt(2.0) * 1e8, 2e-11) ||
        !nearly_equal(qc_offset[0] /
                          (qc_constant[0] * qc_cr_shape[0]),
                      model.qc.offset[0], 2e-12) ||
        !nearly_equal(qc_offset[1] /
                          (qc_constant[1] * qc_cr_shape[1]),
                      model.qc.offset[1], 2e-12)) {
        fprintf(stderr,
                "QC affine cone scaling is inconsistent: "
                "D=[%g,%g], gamma=[%g,%g], c=[%g,%g], "
                "R=[%g,%g], h=[%g,%g], theta=[%g,%g]\n",
                col_scale[0], col_scale[1],
                qc_constant[0], qc_constant[1],
                qc_linear[0], qc_linear[1],
                qc_factor[0], qc_factor[1],
                qc_offset[0], qc_offset[1],
                qc_cr_shape[0], qc_cr_shape[1]);
        goto cleanup;
    }
    ok = 1;

cleanup:
    gpu_workspace_free(&workspace);
    gpu_problem_free(&problem);
    gpu_context_free(&context);
    scaling_free(&scaling);
    model_free(&model);
    return ok ? 0 : 1;
}

int main(void)
{
    HPRQPModel model;
    HPRQPParams params;
    HPRQPScaling scaling;
    HPRQPGPUContext context;
    HPRQPGPUProblem problem;
    HPRQPGPUWorkspace workspace;
    HPRQPStatus status;
    double dual[6] = {0.0, 1.0, 2.0, 0.0, -1.0, 3.0};
    double bt[2] = {0.0, 0.0};
    double kappa[2] = {0.0, 0.0};
    const double dx[2] = {0.2, -0.1};
    const double dw[2] = {0.3, -0.4};
    const double dqw[2] = {0.6, -1.6};
    const double dy[1] = {0.5};
    const double atdy[2] = {0.5, -0.5};
    const double ds[6] = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6};
    const double zeros_n[2] = {0.0, 0.0};
    const double zeros_m[1] = {0.0};
    const double zeros_s[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double btds[2] = {0.0, 0.0};
    double metric = 0.0;
    double expected_metric_sq;
    double sigma_before;
    double expected_sigma;
    HPRQPRestart restart_info;
    HPRQPResiduals residuals;
    const char *path = HPRQP_SOURCE_DIR "/tests/data/simple_qcqp.mps";
    int exit_code = 1;

    if (!gpu_backend_available()) {
        printf("test_gpu_qcqp skipped\n");
        return 0;
    }

    model_init(&model);
    params_set_defaults(&params);
    scaling_init(&scaling);
    gpu_context_init(&context);
    gpu_problem_init(&problem);
    gpu_workspace_init(&workspace);
    restart_init(&restart_info);
    residuals_init(&residuals);

    params.use_gpu = 1;
    params.use_scaling = 0;
    params.method = HPRQP_METHOD_PARALLEL;
    params.parallel_majorizer_mode = HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER;
    params.verbose = 0;
    /* This fixture checks the legacy policy before explicitly enabling the
     * SOCP policy below; do not inherit the public policy default. */
    params.use_socp_sigma_policy = 0;
    params.tau_mode = HPRQP_TAU_MODE_UNIT;

    status = build_from_mps(path, &model);
    if (status != HPRQP_STATUS_OK || model.qc.count != 2) {
        fprintf(stderr, "failed to build QCQP fixture: %d\n", status);
        goto cleanup;
    }
    status = gpu_context_open(&context, 0);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_problem_upload(&problem, &model);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_workspace_allocate(&workspace, &problem, &params);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_problem_scale(&problem, &workspace, &params, &scaling);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_workspace_prepare_spmv(&workspace, &problem, 0);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_workspace_setup(&workspace, &problem, &params);
    if (status != HPRQP_STATUS_OK) goto cleanup;

    {
        const double residual_probe_x[2] = {2.0, -3.0};
        double pairwise_qc[2] = {0.0, 0.0};
        double joint_qc[2] = {0.0, 0.0};

        status = gpu_vector_copy_from_host(&workspace.x_bar,
                                           residual_probe_x, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        workspace.qcqp_scalar_f = 0;
        status = gpu_qcqp_compute_residual_vectors(&workspace, &problem);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_to_host(&workspace.qc_primal_violation,
                                         pairwise_qc, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;

        workspace.qcqp_scalar_f = 1;
        status = gpu_qcqp_compute_residual_vectors(&workspace, &problem);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_to_host(&workspace.qc_primal_violation,
                                         joint_qc, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        workspace.qcqp_scalar_f = 0;
        status = gpu_vector_copy_from_host(&workspace.x_bar, zeros_n, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;

        if (!(pairwise_qc[0] > 3.0) ||
            !nearly_equal(pairwise_qc[0], joint_qc[0], 2e-13) ||
            !nearly_equal(pairwise_qc[1], joint_qc[1], 2e-13)) {
            fprintf(stderr,
                    "QC residual changed with the majorizer: "
                    "pairwise=[%g,%g], joint=[%g,%g]\n",
                    pairwise_qc[0], pairwise_qc[1],
                    joint_qc[0], joint_qc[1]);
            goto cleanup;
        }
    }

    status = gpu_vector_copy_from_host(&workspace.qc_dual, dual, 6);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_qcqp_apply_bt(&workspace, &problem,
                               workspace.qc_dual.values,
                               workspace.BTs.values);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_to_host(&workspace.BTs, bt, 2);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    if (!nearly_equal(bt[0], 3.0, 1e-12) ||
        !nearly_equal(bt[1], 4.0 + 6.0 * sqrt(2.0), 1e-12)) {
        fprintf(stderr, "B^T application is wrong: [%g, %g]\n", bt[0], bt[1]);
        goto cleanup;
    }

    status = gpu_vector_copy_to_host(&workspace.qc_kappa, kappa, 2);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    if (!(isfinite(kappa[0]) && isfinite(kappa[1]) &&
          kappa[0] > 0.0 && kappa[1] > 0.0)) {
        fprintf(stderr, "invalid QC majorizer: [%g, %g]\n", kappa[0], kappa[1]);
        goto cleanup;
    }
    {
        /* lambda_max(sum_i (Q_i + c_i c_i^T) / kappa_i) <= 1. */
        const double a = 8.0 / kappa[0] + 1.0 / kappa[1];
        const double b = -2.0 / kappa[0] + 3.0 / kappa[1];
        const double d = 1.0 / kappa[0] + 17.0 / kappa[1];
        const double lambda = 0.5 * (a + d + sqrt((a - d) * (a - d) + 4.0 * b * b));
        const double expected_ab = sqrt(
            fmin(1.0, params.eig_factor * (a + d - 2.0 * b) /
                          workspace.lambda_max_A));
        const double trace_sq = 2.0 * a + 4.0 * d;
        const double determinant_sq = 8.0 * (a * d - b * b);
        const double lambda_bq = 0.5 *
            (trace_sq + sqrt(trace_sq * trace_sq -
                             4.0 * determinant_sq));
        const double expected_bq = sqrt(
            fmin(1.0, params.eig_factor * lambda_bq /
                          workspace.lambda_max_Q));
        if (!(lambda <= 1.0 + 1e-10)) {
            fprintf(stderr, "D_B is not a valid majorizer: lambda=%g\n", lambda);
            goto cleanup;
        }
        if (!nearly_equal(workspace.zeta_AB, expected_ab, 2e-4) ||
            !nearly_equal(workspace.zeta_BQ, expected_bq, 2e-4) ||
            !(workspace.zeta_AB < 1.0 && workspace.zeta_BQ < 1.0)) {
            fprintf(stderr,
                    "pairwise power estimates are wrong: "
                    "zeta_AB=%g expected=%g, zeta_BQ=%g expected=%g\n",
                    workspace.zeta_AB, expected_ab,
                    workspace.zeta_BQ, expected_bq);
            goto cleanup;
        }
    }

    if (!(workspace.d_y >= 1.0 && workspace.d_s >= 1.0 &&
          workspace.d_w >= 1.0)) {
        fprintf(stderr, "invalid pairwise coefficients\n");
        goto cleanup;
    }

    /*
     * The QCQP restart norm is the QP M-norm with the additional
     * sigma*d_s*||ds||^2_{D_B} and 2<B^T ds,dx> terms.
     */
    status = gpu_vector_copy_from_host(&workspace.dx, dx, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_from_host(&workspace.dw, dw, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_from_host(&workspace.dQw, dqw, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_from_host(&workspace.dy, dy, 1U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_from_host(&workspace.ATdy, atdy, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_from_host(&workspace.qc_dual_delta, ds, 6U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_qcqp_compute_metric_energy(
        &workspace, &problem, workspace.qc_dual_delta.values);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_qcqp_apply_bt(&workspace, &problem,
                               workspace.qc_dual_delta.values,
                               workspace.BTds.values);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_to_host(&workspace.BTds, btds, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_compute_m_norm_parallel_value(&workspace, &metric);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    {
        const double qc_energy = 0.14 * kappa[0] + 0.77 * kappa[1];
        const double dw_dqw = dw[0] * dqw[0] + dw[1] * dqw[1];
        const double atdy_dx = atdy[0] * dx[0] + atdy[1] * dx[1];
        const double btds_dx = btds[0] * dx[0] + btds[1] * dx[1];
        const double dqw_dx = dqw[0] * dx[0] + dqw[1] * dx[1];
        const double dx_sq = dx[0] * dx[0] + dx[1] * dx[1];
        expected_metric_sq =
            workspace.sigma *
                (workspace.d_y * workspace.lambda_max_A * dy[0] * dy[0] +
                 workspace.d_s * qc_energy +
                 workspace.d_w * workspace.lambda_max_Q * dw_dqw) +
            2.0 * (atdy_dx + btds_dx - dqw_dx) +
            dx_sq / workspace.sigma;
        if (!nearly_equal(metric, sqrt(expected_metric_sq), 2e-12)) {
            fprintf(stderr,
                    "QCQP restart M-norm is wrong: %g expected %g\n",
                    metric, sqrt(expected_metric_sq));
            goto cleanup;
        }

        /* The sigma update minimizes sigma*a_M + ||dx||^2/sigma. */
        status = gpu_vector_copy_from_host(&workspace.x_bar, dx, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.last_x, zeros_n, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.w_bar, dw, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.last_w, zeros_n, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.Qw_bar, dqw, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.last_Qw, zeros_n, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.y_bar, dy, 1U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.last_y, zeros_m, 1U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.ATy_bar, atdy, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.last_ATy, zeros_n, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.qc_dual_bar, ds, 6U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.last_qc_dual,
                                           zeros_s, 6U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.BTs_bar, btds, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.last_BTs, zeros_n, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;

        sigma_before = workspace.sigma;
        restart_info.restart_flag = 1;
        restart_info.current_gap = 1.0;
        restart_info.best_gap = 1.0;
        restart_info.best_sigma = sigma_before;
        residuals.err_Rp_org_bar = 0.1;
        residuals.err_QC_org_bar = 0.2;
        residuals.err_Rd_org_bar = 0.15;
        residuals.rel_gap_bar = 0.05;
        residuals.KKTx_and_gap_org_bar = 0.2;
        {
            const double a_m =
                workspace.d_y * workspace.lambda_max_A * dy[0] * dy[0] +
                workspace.d_s * qc_energy +
                workspace.d_w * workspace.lambda_max_Q * dw_dqw;
            const double sigma_estimation = sqrt(dx_sq / a_m);
            const double blend = exp(-0.05);
            expected_sigma = exp(blend * log(sigma_estimation) +
                                 (1.0 - blend) * log(sigma_before));
            expected_sigma = fmin(10.0 * sigma_before,
                                  fmax(0.1 * sigma_before,
                                       expected_sigma));
        }
        status = gpu_parallel_update_metric_value(
            &workspace, &problem, &params, &restart_info, &residuals);
        if (status != HPRQP_STATUS_OK ||
            !nearly_equal(workspace.sigma, expected_sigma, 2e-12) ||
            !restart_info.metric_reseed_pending) {
            fprintf(stderr,
                    "QCQP sigma extension is wrong: sigma=%g expected=%g "
                    "reseed=%d status=%d\n",
                    workspace.sigma, expected_sigma,
                    restart_info.metric_reseed_pending, status);
            goto cleanup;
        }

        /* A single QCQP sigma update cannot move by more than one decade. */
        {
            restart_info.current_gap = 100.0;
            restart_info.best_gap = 1.0;
            residuals.KKTx_and_gap_org_bar = 0.2;
            residuals.is_updated = 1;

            workspace.sigma = 1.0;
            restart_info.best_sigma = 1e-30;
            restart_info.metric_reseed_pending = 0;
            status = gpu_parallel_update_metric_value(
                &workspace, &problem, &params, &restart_info, &residuals);
            if (status != HPRQP_STATUS_OK ||
                !nearly_equal(workspace.sigma, 0.1, 2e-12) ||
                !restart_info.metric_reseed_pending) {
                fprintf(stderr,
                        "QCQP lower sigma safeguard is wrong: "
                        "sigma=%g expected=0.1 reseed=%d status=%d\n",
                        workspace.sigma,
                        restart_info.metric_reseed_pending, status);
                goto cleanup;
            }

            workspace.sigma = 1.0;
            restart_info.best_sigma = 1e30;
            restart_info.metric_reseed_pending = 0;
            status = gpu_parallel_update_metric_value(
                &workspace, &problem, &params, &restart_info, &residuals);
            if (status != HPRQP_STATUS_OK ||
                !nearly_equal(workspace.sigma, 10.0, 2e-12) ||
                !restart_info.metric_reseed_pending) {
                fprintf(stderr,
                        "QCQP upper sigma safeguard is wrong: "
                        "sigma=%g expected=10 reseed=%d status=%d\n",
                        workspace.sigma,
                        restart_info.metric_reseed_pending, status);
                goto cleanup;
            }

            /* The active HPR-SOCP sigma policy obeys the same safeguard. */
            params.use_socp_sigma_policy = 1;
            residuals.err_Rp_org_bar = 0.1;
            residuals.err_QC_org_bar = 0.1;
            residuals.err_Rd_org_bar = 0.1;
            residuals.rel_gap_bar = 0.1;
            residuals.KKTx_and_gap_org_bar = 2.0;
            restart_info.current_gap = 10.0;
            restart_info.best_gap = 1.0;
            restart_info.best_kkt = 0.2;
            restart_info.sigma_correction_active = 0;
            restart_info.sigma_correction_hold = -1;
            restart_info.sigma_correction_dir = 0;

            workspace.sigma = 1.0;
            restart_info.best_sigma = 1e30;
            restart_info.metric_reseed_pending = 0;
            status = gpu_parallel_update_metric_value(
                &workspace, &problem, &params, &restart_info, &residuals);
            if (status != HPRQP_STATUS_OK ||
                !nearly_equal(workspace.sigma, 10.0, 2e-12) ||
                !restart_info.metric_reseed_pending) {
                fprintf(stderr,
                        "SOCP upper sigma safeguard is wrong: "
                        "sigma=%g expected=10 reseed=%d status=%d\n",
                        workspace.sigma,
                        restart_info.metric_reseed_pending, status);
                goto cleanup;
            }

            workspace.sigma = 1.0;
            restart_info.best_sigma = 1e-30;
            restart_info.metric_reseed_pending = 0;
            status = gpu_parallel_update_metric_value(
                &workspace, &problem, &params, &restart_info, &residuals);
            if (status != HPRQP_STATUS_OK ||
                !nearly_equal(workspace.sigma, 0.1, 2e-12) ||
                !restart_info.metric_reseed_pending) {
                fprintf(stderr,
                        "SOCP lower sigma safeguard is wrong: "
                        "sigma=%g expected=0.1 reseed=%d status=%d\n",
                        workspace.sigma,
                        restart_info.metric_reseed_pending, status);
                goto cleanup;
            }
            params.use_socp_sigma_policy = 0;
        }

        /* Each QCQP coupling edge gets its own restart-adaptive shape. */
        {
            const double energy_y = workspace.lambda_max_A * dy[0] * dy[0];
            const double energy_s = qc_energy;
            const double energy_w = workspace.lambda_max_Q * dw_dqw;
            const double expected_tau_ab = sqrt(energy_s / energy_y);
            const double expected_tau_aq = sqrt(energy_w / energy_y);
            const double expected_tau_bq = sqrt(energy_w / energy_s);
            double expected_d_y;
            double expected_d_s;
            double expected_d_w;

            params.tau_mode = HPRQP_TAU_MODE_ADAPTIVE_RESTART;
            params.sigma_fixed = 1;
            restart_info.metric_reseed_pending = 0;
            status = gpu_parallel_update_metric_value(
                &workspace, &problem, &params, &restart_info, &residuals);
            expected_d_y = 1.0 +
                workspace.zeta_AB * expected_tau_ab +
                workspace.zeta_AQ * expected_tau_aq;
            expected_d_s = 1.0 +
                workspace.zeta_AB / expected_tau_ab +
                workspace.zeta_BQ * expected_tau_bq;
            expected_d_w = 1.0 +
                workspace.zeta_AQ / expected_tau_aq +
                workspace.zeta_BQ / expected_tau_bq;
            if (status != HPRQP_STATUS_OK ||
                !nearly_equal(workspace.tau_AB, expected_tau_ab, 2e-12) ||
                !nearly_equal(workspace.tau_AQ, expected_tau_aq, 2e-12) ||
                !nearly_equal(workspace.tau_BQ, expected_tau_bq, 2e-12) ||
                !nearly_equal(workspace.d_y, expected_d_y, 2e-12) ||
                !nearly_equal(workspace.d_s, expected_d_s, 2e-12) ||
                !nearly_equal(workspace.d_w, expected_d_w, 2e-12) ||
                !restart_info.metric_reseed_pending) {
                fprintf(stderr,
                        "QCQP adaptive pairwise shapes are wrong: "
                        "tau=[%g,%g,%g], d=[%g,%g,%g], status=%d\n",
                        workspace.tau_AB, workspace.tau_AQ,
                        workspace.tau_BQ, workspace.d_y,
                        workspace.d_s, workspace.d_w, status);
                goto cleanup;
            }
        }

        /* The adaptive A-B shape uses the configured bounds.  Exercise both
         * saturation directions without changing the A-Q or B-Q bounds. */
        {
            const double dy_small[1] = {1e-5};
            double ds_small[6];

            params.tau_min = 1.0 / 40.0;
            params.tau_max = 40.0;
            for (size_t i = 0U; i < 6U; ++i) {
                ds_small[i] = 1e-5 * ds[i];
            }

            status = gpu_vector_copy_from_host(&workspace.dy, dy_small, 1U);
            if (status != HPRQP_STATUS_OK) goto cleanup;
            restart_info.metric_reseed_pending = 0;
            status = gpu_parallel_update_metric_value(
                &workspace, &problem, &params, &restart_info, &residuals);
            if (status != HPRQP_STATUS_OK ||
                !nearly_equal(workspace.tau_AB, params.tau_max, 2e-12) ||
                !restart_info.metric_reseed_pending) {
                fprintf(stderr,
                        "QCQP adaptive tau_AB upper cap is wrong: "
                        "tau=%g expected=%g reseed=%d status=%d\n",
                        workspace.tau_AB, params.tau_max,
                        restart_info.metric_reseed_pending, status);
                goto cleanup;
            }

            status = gpu_vector_copy_from_host(&workspace.dy, dy, 1U);
            if (status != HPRQP_STATUS_OK) goto cleanup;
            status = gpu_vector_copy_from_host(&workspace.qc_dual_delta,
                                               ds_small, 6U);
            if (status != HPRQP_STATUS_OK) goto cleanup;
            restart_info.metric_reseed_pending = 0;
            status = gpu_parallel_update_metric_value(
                &workspace, &problem, &params, &restart_info, &residuals);
            if (status != HPRQP_STATUS_OK ||
                !nearly_equal(workspace.tau_AB, params.tau_min, 2e-12) ||
                !restart_info.metric_reseed_pending) {
                fprintf(stderr,
                        "QCQP adaptive tau_AB lower cap is wrong: "
                        "tau=%g expected=%g reseed=%d status=%d\n",
                        workspace.tau_AB, params.tau_min,
                        restart_info.metric_reseed_pending, status);
                goto cleanup;
            }

            status = gpu_vector_copy_from_host(&workspace.qc_dual_delta,
                                               ds, 6U);
            if (status != HPRQP_STATUS_OK) goto cleanup;
        }

    }

    /* A QCQP restart must move both s and B^T s into the new anchor. */
    status = gpu_restart_apply_parallel_value(&workspace);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_to_host(&workspace.last_qc_dual, dual, 6U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    status = gpu_vector_copy_to_host(&workspace.last_BTs, bt, 2U);
    if (status != HPRQP_STATUS_OK) goto cleanup;
    for (size_t i = 0U; i < 6U; ++i) {
        if (!nearly_equal(dual[i], ds[i], 1e-13)) {
            fprintf(stderr, "QCQP restart did not copy cone dual anchor\n");
            goto cleanup;
        }
    }
    if (!nearly_equal(bt[0], btds[0], 1e-13) ||
        !nearly_equal(bt[1], btds[1], 1e-13)) {
        fprintf(stderr, "QCQP restart did not copy B^T s anchor\n");
        goto cleanup;
    }

    /* The explicit joint-F/Q path uses one scalar metric on [A;B], one
     * F-Q coupling, and one restart-adaptive shape.  Under the MPS
     * QCMATRIX convention, this fixture has F^*F=diag(10,19) and
     * Q=diag(2,4), so lambda_max(FQF^*)=76. */
    {
        const double expected_lambda_f = params.eig_factor * 19.0;
        const double expected_zeta_fq = sqrt(
            params.eig_factor * 76.0 /
            (expected_lambda_f * workspace.lambda_max_Q));
        const double ds_sq = 0.91;
        const double dy_sq = dy[0] * dy[0];
        const double dw_dqw = dw[0] * dqw[0] + dw[1] * dqw[1];
        double expected_tau_fq;
        double expected_d_f;
        double expected_d_w;

        params.qc_majorizer_mode = HPRQP_QC_MAJORIZER_JOINT_F;
        params.tau_mode = HPRQP_TAU_MODE_ADAPTIVE_RESTART;
        params.sigma_fixed = 1;
        workspace.tau = 1.0;
        status = gpu_qcqp_setup_majorizer(&workspace, &problem, &params);
        if (status != HPRQP_STATUS_OK || !workspace.qcqp_scalar_f ||
            !nearly_equal(workspace.lambda_F, expected_lambda_f, 2e-4) ||
            !nearly_equal(workspace.zeta_AQ, expected_zeta_fq, 2e-4) ||
            !nearly_equal(workspace.zeta_AB, 0.0, 2e-13) ||
            !nearly_equal(workspace.zeta_BQ, 0.0, 2e-13) ||
            !nearly_equal(workspace.d_y, workspace.d_s, 2e-13)) {
            fprintf(stderr,
                    "joint-F/Q setup is wrong: lambda_F=%g expected=%g, "
                    "zeta_FQ=%g expected=%g, d=[%g,%g,%g], status=%d\n",
                    workspace.lambda_F, expected_lambda_f,
                    workspace.zeta_AQ, expected_zeta_fq,
                    workspace.d_y, workspace.d_s, workspace.d_w, status);
            goto cleanup;
        }

        status = gpu_vector_copy_from_host(&workspace.dy, dy, 1U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.dw, dw, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.dQw, dqw, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_from_host(&workspace.qc_dual_delta,
                                           ds, 6U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        expected_tau_fq = sqrt(
            workspace.lambda_max_Q * dw_dqw /
            (workspace.lambda_F * (dy_sq + ds_sq)));
        expected_d_f = 1.0 + workspace.zeta_AQ * expected_tau_fq;
        expected_d_w = 1.0 + workspace.zeta_AQ / expected_tau_fq;
        restart_info.restart_flag = 1;
        restart_info.metric_reseed_pending = 0;
        status = gpu_parallel_update_metric_value(
            &workspace, &problem, &params, &restart_info, &residuals);
        if (status != HPRQP_STATUS_OK ||
            !nearly_equal(workspace.tau_AQ, expected_tau_fq, 2e-12) ||
            !nearly_equal(workspace.d_y, expected_d_f, 2e-12) ||
            !nearly_equal(workspace.d_s, expected_d_f, 2e-12) ||
            !nearly_equal(workspace.d_w, expected_d_w, 2e-12) ||
            !restart_info.metric_reseed_pending) {
            fprintf(stderr,
                    "joint-F/Q adaptive shape is wrong: tau=%g expected=%g, "
                    "d=[%g,%g,%g] expected=[%g,%g,%g], reseed=%d, "
                    "status=%d\n",
                    workspace.tau_AQ, expected_tau_fq,
                    workspace.d_y, workspace.d_s, workspace.d_w,
                    expected_d_f, expected_d_f, expected_d_w,
                    restart_info.metric_reseed_pending, status);
            goto cleanup;
        }
    }

    /* The scalar-B path keeps A, B, and Q as three pairwise groups but uses
     * D_B=lambda_B I.  For this fixture,
     * B^*B=[[9,1],[1,18]], so lambda_max(B^*B)=(27+sqrt(85))/2. */
    {
        const double raw_lambda_b = 0.5 * (27.0 + sqrt(85.0));
        const double expected_lambda_b = params.eig_factor * raw_lambda_b;
        const double expected_ab = sqrt(
            params.eig_factor * 25.0 /
            (expected_lambda_b * workspace.lambda_max_A));
        const double raw_lambda_bq = 45.0 + sqrt(737.0);
        const double expected_bq = sqrt(
            params.eig_factor * raw_lambda_bq /
            (expected_lambda_b * workspace.lambda_max_Q));
        double scalar_kappa[2] = {0.0, 0.0};

        params.qc_majorizer_mode = HPRQP_QC_MAJORIZER_SCALAR_B;
        params.tau_mode = HPRQP_TAU_MODE_ADAPTIVE_RESTART;
        workspace.tau = 1.0;
        status = gpu_qcqp_setup_majorizer(&workspace, &problem, &params);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        status = gpu_vector_copy_to_host(&workspace.qc_kappa,
                                         scalar_kappa, 2U);
        if (status != HPRQP_STATUS_OK) goto cleanup;
        if (workspace.qcqp_scalar_f ||
            !nearly_equal(workspace.lambda_F, 0.0, 2e-13) ||
            !nearly_equal(scalar_kappa[0], expected_lambda_b, 2e-4) ||
            !nearly_equal(scalar_kappa[1], expected_lambda_b, 2e-4) ||
            !nearly_equal(workspace.zeta_AB, expected_ab, 2e-4) ||
            !nearly_equal(workspace.zeta_BQ, expected_bq, 2e-4) ||
            !(workspace.d_y >= 1.0 && workspace.d_s >= 1.0 &&
              workspace.d_w >= 1.0)) {
            fprintf(stderr,
                    "scalar-B setup is wrong: kappa=[%g,%g] expected=%g, "
                    "zeta_AB=%g expected=%g, zeta_BQ=%g expected=%g, "
                    "d=[%g,%g,%g], status=%d\n",
                    scalar_kappa[0], scalar_kappa[1], expected_lambda_b,
                    workspace.zeta_AB, expected_ab,
                    workspace.zeta_BQ, expected_bq,
                    workspace.d_y, workspace.d_s, workspace.d_w, status);
            goto cleanup;
        }
    }

    if (test_qc_affine_cone_scaling() != 0) goto cleanup;
    if (test_active_qcqp_solver() != 0) goto cleanup;

    printf("test_gpu_qcqp passed: kappa=[%.12g, %.12g], "
           "zeta=[%.6g, %.6g, %.6g], d=[%.6g, %.6g, %.6g]\n",
           kappa[0], kappa[1], workspace.zeta_AB, workspace.zeta_AQ,
           workspace.zeta_BQ, workspace.d_y, workspace.d_s,
           workspace.d_w);
    exit_code = 0;

cleanup:
    gpu_workspace_free(&workspace);
    gpu_problem_free(&problem);
    gpu_context_free(&context);
    scaling_free(&scaling);
    model_free(&model);
    return exit_code;
}
