#ifndef HPRQP_GPU_TYPES_H
#define HPRQP_GPU_TYPES_H

#include <stddef.h>
#include <stdint.h>

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HPRQPGPUContext {
    int requested_device;
    int initialized;
} HPRQPGPUContext;

typedef struct HPRQPGPUVector {
    size_t length;
    double *values;
    int allocated;
} HPRQPGPUVector;

typedef int32_t HPRQPGPUCsrIndex;

typedef enum HPRQPGPUCustomSpmvKernel {
    HPRQP_GPU_CUSTOM_SPMV_ROW = 0,
    HPRQP_GPU_CUSTOM_SPMV_WARP = 1
} HPRQPGPUCustomSpmvKernel;

typedef enum HPRQPGPUParallelPersistentKernel {
    HPRQP_GPU_PARALLEL_PERSISTENT_ROW = 0,
    HPRQP_GPU_PARALLEL_PERSISTENT_ROW_PIPELINE = 1,
    HPRQP_GPU_PARALLEL_PERSISTENT_GROUP = 2
} HPRQPGPUParallelPersistentKernel;

#define HPRQP_GPU_PARALLEL_BATCH_MAX 128U

typedef struct HPRQPGPUSpMVA {
    void *handle;
    void *desc_A;
    void *desc_x_bar;
    void *desc_x_hat;
    void *desc_dx;
    void *desc_tempv;
    void *desc_Ax;
    void *buffer;
    size_t buffer_size;
    void *graph_x_hat;
    void *graph_exec_x_hat;
    int prepared;
} HPRQPGPUSpMVA;

typedef struct HPRQPGPUSpMVAT {
    void *handle;
    void *desc_AT;
    void *desc_y_bar;
    void *desc_y;
    void *desc_ATy_bar;
    void *desc_ATy;
    void *desc_ATdy;
    void *buffer;
    size_t buffer_size;
    void *graph_y;
    void *graph_exec_y;
    int prepared;
} HPRQPGPUSpMVAT;

typedef struct HPRQPGPUSpMVQ {
    void *handle;
    void *desc_Q;
    void *desc_w;
    void *desc_w_bar;
    void *desc_x_bar;
    void *desc_Qw;
    void *desc_Qw_bar;
    void *desc_Qx;
    void *buffer;
    size_t buffer_size;
    int prepared;
} HPRQPGPUSpMVQ;

typedef struct HPRQPGPUMatrixCSR {
    size_t nrow;
    size_t ncol;
    size_t nnz;
    HPRQPGPUCsrIndex *row_ptr;
    HPRQPGPUCsrIndex *col_ind;
    double *values;
    int uploaded;
} HPRQPGPUMatrixCSR;

typedef struct HPRQPGPUProblem {
    size_t m;
    size_t n;
    size_t p;
    size_t qc_cone_dim;
    size_t nnz_q;
    size_t nnz_a;
    size_t nnz_qc_linear;
    size_t nnz_qc_factor;
    int empty_A;
    int disjoint_quadratic_support;
    double obj_constant;
    HPRQPGPUMatrixCSR Q;
    HPRQPGPUMatrixCSR A;
    HPRQPGPUMatrixCSR AT;
    HPRQPGPUMatrixCSR qc_linear;
    /* Thin diagonal specialization of R_i, not diag(Q_i).  Row i stores
     * exactly r_i factor entries, so cone block i has dimension r_i + 2. */
    HPRQPGPUMatrixCSR qc_factor;
    HPRQPGPUVector c;
    HPRQPGPUVector AL;
    HPRQPGPUVector AU;
    HPRQPGPUVector l;
    HPRQPGPUVector u;
    HPRQPGPUVector qc_constant;
    HPRQPGPUVector qc_offset;
    int uploaded;
} HPRQPGPUProblem;

typedef struct HPRQPGPUWorkspace {
    size_t m;
    size_t n;
    size_t p;
    size_t qc_cone_dim;
    int empty_A;
    unsigned int threads_n;
    unsigned int blocks_n;
    unsigned int threads_m;
    unsigned int blocks_m;
    unsigned int threads_p;
    unsigned int blocks_p;
    HPRQPSpmvMode spmv_mode_q;
    HPRQPSpmvMode spmv_mode_a;
    HPRQPGPUCustomSpmvKernel custom_spmv_q_kernel;
    HPRQPGPUCustomSpmvKernel custom_spmv_a_kernel;
    int parallel_fuse_q;
    int parallel_fuse_at;
    int parallel_fuse_a;
    int parallel_persistent_enabled;
    HPRQPGPUParallelPersistentKernel parallel_persistent_kernel;
    unsigned int parallel_persistent_threads;
    unsigned int parallel_persistent_blocks;
    HPRQPGPUCsrIndex *parallel_q_nonempty_rows;
    size_t parallel_q_nonempty_count;
    int parallel_q_exact_single_diagonal;
    uint16_t *parallel_q_col16;
    uint16_t *parallel_a_col16;
    uint16_t *parallel_at_col16;
    HPRQPGPUCsrIndex *parallel_q_col16_base;
    HPRQPGPUCsrIndex *parallel_a_col16_base;
    HPRQPGPUCsrIndex *parallel_at_col16_base;
    HPRQPGPUCsrIndex *qc_bt_row_ptr;
    HPRQPGPUCsrIndex *qc_bt_dual_ind;
    double *qc_bt_values;
    size_t qc_bt_nnz;
    int qc_bt_short_rows;
    int parallel_col16_available;
    int parallel_col16_delta_available;
    int parallel_persistent_use_col16;
    unsigned int parallel_persistent_fixed_row_length;
    int parallel_x_lower_positive_zero;
    int parallel_constraints_positive_zero_equality;
    unsigned int parallel_persistent_q_group_width;
    double sigma;
    double lambda_max_A;
    double lambda_max_Q;
    double lambda_max_AQA;
    double lambda_max_AQ2A;
    double zeta_AQ;
    double tau;
    double tau_initial;
    double d_y;
    double d_s;
    double d_w;
    double zeta_AB;
    double zeta_BQ;
    double tau_AB;
    double tau_AQ;
    double tau_BQ;
    double lambda_F;
    int qcqp_scalar_f;
    int qcqp_diag_bq;
    double qcqp_diag_bq_base_s;
    double qcqp_diag_bq_base_w;
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
    double scale_b_scale;
    double scale_c_scale;
    double scale_norm_b_org;
    double scale_norm_c_org;
    HPRQPGPUVector x;
    HPRQPGPUVector x_bar;
    HPRQPGPUVector x_hat;
    HPRQPGPUVector z_bar;
    HPRQPGPUVector dx;
    HPRQPGPUVector w;
    HPRQPGPUVector w_bar;
    HPRQPGPUVector dw;
    HPRQPGPUVector Qw;
    HPRQPGPUVector Qw_bar;
    HPRQPGPUVector Qx;
    HPRQPGPUVector y;
    HPRQPGPUVector y_bar;
    HPRQPGPUVector dy;
    HPRQPGPUVector s;
    HPRQPGPUVector Rp;
    HPRQPGPUVector Rd;
    HPRQPGPUVector Ax;
    HPRQPGPUVector ATy;
    HPRQPGPUVector ATy_bar;
    HPRQPGPUVector ATdy;
    HPRQPGPUVector dQw;
    HPRQPGPUVector QATdy;
    HPRQPGPUVector tempv;
    HPRQPGPUVector last_x;
    HPRQPGPUVector last_y;
    HPRQPGPUVector last_w;
    HPRQPGPUVector last_Qw;
    HPRQPGPUVector last_ATy;
    HPRQPGPUVector fact1;
    HPRQPGPUVector fact2;
    HPRQPGPUVector fact;
    HPRQPGPUVector fact_M;
    HPRQPGPUVector diag_Q;
    HPRQPGPUVector qc_dual;
    HPRQPGPUVector qc_dual_bar;
    HPRQPGPUVector qc_dual_delta;
    HPRQPGPUVector last_qc_dual;
    HPRQPGPUVector BTs;
    HPRQPGPUVector BTs_bar;
    HPRQPGPUVector BTds;
    HPRQPGPUVector last_BTs;
    HPRQPGPUVector qc_kappa;
    HPRQPGPUVector qc_theta;
    HPRQPGPUVector qc_diag_bq_row_weight;
    HPRQPGPUVector qc_diag_bq_col_weight;
    HPRQPGPUVector qc_diag_bq_cone_metric;
    HPRQPGPUVector qc_diag_bq_w_gamma;
    HPRQPGPUVector qc_diag_bq_w_inv;
    HPRQPGPUVector qc_diag_bq_metric_energy;
    HPRQPGPUVector qc_diag_bq_work;
    HPRQPGPUVector qc_scale_step;
    HPRQPGPUVector qc_cr_shape;
    HPRQPGPUVector qc_incidence_sum;
    HPRQPGPUVector qc_primal_violation;
    HPRQPGPUVector qc_bx_norm;
    HPRQPGPUVector qc_h_norm;
    HPRQPGPUVector qc_dual_objective;
    HPRQPGPUVector qc_scaled_bts_abs;
    HPRQPGPUVector row_norm;
    HPRQPGPUVector col_norm;
    HPRQPGPUVector diag_tmp;
    HPRQPGPUVector reduce_tmp1;
    HPRQPGPUVector reduce_tmp2;
    HPRQPGPUVector batch_halpern;
    HPRQPGPUVector batch_halpern_fact2;
    void *blas_handle;
    void *qc_cone_stream;
    void *qc_x_ready_event;
    void *qc_cone_done_event;
    void *reduce_buffer;
    size_t reduce_buffer_size;
    HPRQPGPUSpMVA spmv_A;
    HPRQPGPUSpMVAT spmv_AT;
    HPRQPGPUSpMVQ spmv_Q;
    int spmv_prepared;
    int allocated;
} HPRQPGPUWorkspace;

#ifdef __cplusplus
}
#endif

#endif
