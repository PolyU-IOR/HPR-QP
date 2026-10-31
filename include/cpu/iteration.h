#ifndef HPRQP_ITERATION_H
#define HPRQP_ITERATION_H

#include <stddef.h>

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus iteration_step(HPRQPWorkspace *workspace, size_t inner_iter);
HPRQPStatus iteration_step_parallel(HPRQPWorkspace *workspace,
                                    size_t inner_iter);

#ifdef __cplusplus
}
#endif

#endif
