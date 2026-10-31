#ifndef HPRQP_H
#define HPRQP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "gpu/gpu.h"
#include "shared/sparse.h"
#include "shared/types.h"

void model_init(HPRQPModel *model);
void model_free(HPRQPModel *model);
HPRQPStatus model_copy(HPRQPModel *dest, const HPRQPModel *src);
HPRQPStatus build_from_QAbc(const HPRQPMatrixCSR *Q,
							const HPRQPMatrixCSR *A,
							const double *c,
							const double *AL,
							const double *AU,
							const double *l,
							const double *u,
							double obj_constant,
							HPRQPModel *model);
HPRQPStatus build_from_mps(const char *mps_path, HPRQPModel *model);

void params_set_defaults(HPRQPParams *params);
void results_init(HPRQPResults *results);
void results_free(HPRQPResults *results);
void print_model_info(const HPRQPModel *model);
void print_params_info(const HPRQPParams *params);
HPRQPStatus solve_gpu(const HPRQPModel *model, const HPRQPParams *params, HPRQPResults *results);
HPRQPStatus solve(const HPRQPModel *model, const HPRQPParams *params, HPRQPResults *results);

#ifdef __cplusplus
}
#endif

#endif
