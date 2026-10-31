#ifndef HPRQP_BUILD_H
#define HPRQP_BUILD_H

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void model_init(HPRQPModel *model);
void model_free(HPRQPModel *model);
HPRQPStatus model_copy(HPRQPModel *dest, const HPRQPModel *src);
HPRQPStatus build_model_from_arrays(const HPRQPMatrixCSR *Q,
                                    const HPRQPMatrixCSR *A,
                                    const double *c,
                                    const double *AL,
                                    const double *AU,
                                    const double *l,
                                    const double *u,
                                    double obj_constant,
                                    HPRQPModel *model);
HPRQPStatus build_model_from_mps(const char *mps_path, HPRQPModel *model);
HPRQPStatus preprocess_model(HPRQPModel *model);

#ifdef __cplusplus
}
#endif

#endif
