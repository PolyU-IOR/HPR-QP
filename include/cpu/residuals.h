#ifndef HPRQP_RESIDUALS_H
#define HPRQP_RESIDUALS_H

#include <stddef.h>

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void residuals_init(HPRQPResiduals *residuals);

HPRQPStatus residuals_compute(HPRQPWorkspace *workspace,
                              const HPRQPModel *model,
                              const HPRQPScaling *scaling,
                              HPRQPResiduals *residuals,
                              size_t iter);
HPRQPStatus residuals_finalize_from_products(HPRQPWorkspace *workspace,
                                             const HPRQPModel *model,
                                             const HPRQPScaling *scaling,
                                             HPRQPResiduals *residuals,
                                             size_t iter);

#ifdef __cplusplus
}
#endif

#endif