#ifndef HPRQP_WORKSPACE_H
#define HPRQP_WORKSPACE_H

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void workspace_init(HPRQPWorkspace *workspace);
void workspace_free(HPRQPWorkspace *workspace);
HPRQPStatus workspace_allocate(const HPRQPModel *model,
                               const HPRQPParams *params,
                               HPRQPWorkspace *workspace);

#ifdef __cplusplus
}
#endif

#endif