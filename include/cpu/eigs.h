#ifndef HPRQP_EIGS_H
#define HPRQP_EIGS_H

#include "shared/types.h"

#ifdef __cplusplus
extern "C" {
#endif

HPRQPStatus estimate_eigenvalues(HPRQPWorkspace *workspace,
                                 const HPRQPParams *params);

#ifdef __cplusplus
}
#endif

#endif
