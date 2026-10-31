#ifndef HPRQP_TYPES_H
#define HPRQP_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum HPRQPStatus {
    HPRQP_STATUS_OK = 0,
    HPRQP_STATUS_INVALID_ARGUMENT = 1,
    HPRQP_STATUS_ALLOCATION_FAILED = 2,
    HPRQP_STATUS_NOT_IMPLEMENTED = 3,
    HPRQP_STATUS_MAX_ITER = 4,
    HPRQP_STATUS_TIME_LIMIT = 5,
    HPRQP_STATUS_OPTIMAL = 6,
    HPRQP_STATUS_CONTINUE = 7
} HPRQPStatus;

typedef enum HPRQPSpmvMode {
    HPRQP_SPMV_MODE_AUTO = 0,
    HPRQP_SPMV_MODE_CUSPARSE = 1,
    HPRQP_SPMV_MODE_CUSTOMIZED = 2
} HPRQPSpmvMode;

typedef enum HPRQPMethod {
    HPRQP_METHOD_SGS = 0,
    HPRQP_METHOD_PARALLEL = 1
} HPRQPMethod;

typedef enum HPRQPParallelMajorizerMode {
    HPRQP_PARALLEL_MAJORIZER_WORST_CASE = 0,
    HPRQP_PARALLEL_MAJORIZER_COUPLING_POWER = 1,
    HPRQP_PARALLEL_MAJORIZER_TWO_MOMENT = 2,
    HPRQP_PARALLEL_MAJORIZER_MOMENT_SELECTOR = 3
} HPRQPParallelMajorizerMode;

typedef enum HPRQPQCMajorizerMode {
    HPRQP_QC_MAJORIZER_AUTO = 0,
    HPRQP_QC_MAJORIZER_PAIRWISE = 1,
    HPRQP_QC_MAJORIZER_JOINT_F = 2,
    HPRQP_QC_MAJORIZER_SCALAR_B = 3,
    HPRQP_QC_MAJORIZER_DIAG_BQ = 4
} HPRQPQCMajorizerMode;

typedef enum HPRQPTauMode {
    HPRQP_TAU_MODE_UNIT = 0,
    HPRQP_TAU_MODE_FIXED = 1,
    HPRQP_TAU_MODE_BALANCED = 2,
    HPRQP_TAU_MODE_ADAPTIVE_RESTART = 3
} HPRQPTauMode;

typedef struct HPRQPMatrixCSR {
    size_t nrow;
    size_t ncol;
    size_t nnz;
    size_t *row_ptr;
    size_t *col_ind;
    double *values;
    int owns_data;
} HPRQPMatrixCSR;

/*
 * Convex quadratic constraints
 *
 *   0.5 x' Q_i x + c_i' x + d_i <= 0,  i = 0, ..., count - 1.
 *
 * The first implementation targets the Hans Mittelmann convex-QCQP data:
 * every Q_i is diagonal positive semidefinite.  `linear` stores the c_i rows
 * and `diagonal` stores diag(Q_i), both as count-by-n CSR matrices.  The
 * rotated-cone representation uses one copy of the data,
 *
 *   (1, -c_i' x - d_i, sqrt(Q_i) x) in Q_r.
 *
 * `cone_ptr` indexes the flattened cone vectors; block i has dimension
 * 2 + nnz(diagonal row i).
 */
typedef struct HPRQPQuadraticConstraints {
    size_t count;
    size_t cone_dim;
    HPRQPMatrixCSR linear;
    HPRQPMatrixCSR diagonal;
    double *offset;
    size_t *cone_ptr;
    int disjoint_quadratic_support;
} HPRQPQuadraticConstraints;

typedef struct HPRQPModel {
    HPRQPMatrixCSR Q;
    HPRQPMatrixCSR A;
    HPRQPMatrixCSR AT;
    double *c;
    double *AL;
    double *AU;
    double *l;
    double *u;
    HPRQPQuadraticConstraints qc;
    double obj_constant;
    int owns_data;
} HPRQPModel;

typedef struct HPRQPScaling {
    double *row_norm;
    double *col_norm;
    double *l_org;
    double *u_org;
    double b_scale;
    double c_scale;
    double norm_b;
    double norm_c;
    double norm_b_org;
    double norm_c_org;
} HPRQPScaling;

typedef struct HPRQPParams {
    double stoptol;
    size_t max_iter;
    int use_gpu;
    int device_number;
    double time_limit;
    size_t check_iter;
    int print_frequency;
    int use_scaling;
    int use_cr_scaling;
    size_t cr_iterations;
    int use_bc_scaling;
    int verbose;
    HPRQPSpmvMode spmv_mode_q;
    HPRQPSpmvMode spmv_mode_a;
    HPRQPMethod method;
    HPRQPParallelMajorizerMode parallel_majorizer_mode;
    HPRQPQCMajorizerMode qc_majorizer_mode;
    HPRQPTauMode tau_mode;
    double tau;
    double tau_min;
    double tau_max;
    double eig_factor;
    double metric_negative_tolerance;
    double parallel_adaptation_energy_floor;
    int sigma_fixed;
    double initial_sigma;
    int use_socp_restart_policy;
    int use_socp_sigma_policy;
} HPRQPParams;

typedef struct HPRQPWorkspace {
    size_t m;
    size_t n;
    double sigma;
    double lambda_max_A;
    double lambda_max_Q;
    double lambda_max_AQA;
    double lambda_max_AQ2A;
    double zeta_AB;
    double zeta_AQ;
    double zeta_BQ;
    double tau_AB;
    double tau_AQ;
    double tau_BQ;
    double tau;
    double tau_initial;
    double d_y;
    double d_s;
    double d_w;
    double parallel_y_rho;
    double parallel_y_inv;
    double parallel_w_gamma;
    double parallel_w_inv;
    double metric_negative_tolerance;
    size_t metric_reseeds;
    HPRQPMethod method;
    HPRQPParallelMajorizerMode parallel_majorizer_mode;
    int Q_is_diag;
    int eigenvalues_initialized;
    int noC;
    int to_check;
    const HPRQPMatrixCSR *Q;
    const HPRQPMatrixCSR *A;
    const HPRQPMatrixCSR *AT;
    const double *c;
    const double *AL;
    const double *AU;
    const double *l;
    const double *u;
    double *x;
    double *x_bar;
    double *x_hat;
    double *w;
    double *w_bar;
    double *w_hat;
    double *y;
    double *y_bar;
    double *y_hat;
    double *Ax;
    double *ATy;
    double *ATy_bar;
    double *ATdy;
    double *Qw;
    double *Qw_bar;
    double *Qx;
    double *dQw;
    double *QATdy;
    double *dx;
    double *dy;
    double *dw;
    double *Rp;
    double *Rd;
    double *s;
    double *z_bar;
    double *tempv;
    double *diag_Q;
    double *fact1;
    double *fact2;
    double *fact;
    double *fact_M;
    double *last_x;
    double *last_y;
    double *last_Qw;
    double *last_w;
    double *last_ATy;
} HPRQPWorkspace;

typedef struct HPRQPResiduals {
    int is_updated;
    double err_Rp_org_bar;
    double err_QC_org_bar;
    double err_Rd_org_bar;
    double KKTx_and_gap_org_bar;
    double primal_obj_bar;
    double rel_gap_bar;
    double dual_obj_bar;
} HPRQPResiduals;

typedef struct HPRQPRestart {
    int restart_flag;
    int first_restart;
    double last_gap;
    double current_gap;
    double save_gap;
    size_t inner;
    size_t step;
    size_t sufficient;
    size_t necessary;
    size_t long_restart;
    size_t ratio;
    size_t times;
    double weighted_norm;
    double best_gap;
    double best_kkt;
    double best_sigma;
    size_t best_iter;
    int sigma_correction_active;
    int sigma_correction_hold;
    int sigma_correction_dir;
    int metric_reseed_pending;
} HPRQPRestart;

typedef struct HPRQPResults {
    size_t iter;
    size_t iter_4;
    size_t iter_6;
    double time;
    double time_4;
    double time_6;
    double primal_obj;
    double residuals;
    double primal_residual;
    double qc_residual;
    double dual_residual;
    double gap;
    double lambda_max_A;
    double lambda_max_Q;
    double lambda_max_AQA;
    double lambda_max_AQ2A;
    double zeta_AB;
    double zeta_AQ;
    double zeta_BQ;
    double tau_AB;
    double tau_AQ;
    double tau_BQ;
    double tau;
    double d_y;
    double d_s;
    double d_w;
    HPRQPMethod method;
    int status;
    double *x;
    double *y;
    double *z;
    double *w;
    size_t qc_cone_dim;
    double *qc_dual;
} HPRQPResults;

#ifdef __cplusplus
}
#endif

#endif
