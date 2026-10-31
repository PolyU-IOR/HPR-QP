#include "hprqp.h"

#include <stdio.h>

HPRQPStatus solve_gpu(const HPRQPModel *model, const HPRQPParams *params, HPRQPResults *results)
{
    HPRQPParams cpu_params;

    if (model == NULL || params == NULL || results == NULL) {
        return HPRQP_STATUS_INVALID_ARGUMENT;
    }

    cpu_params = *params;
    cpu_params.use_gpu = 0;

    if (params->verbose) {
        fprintf(stderr,
                "GPU backend unavailable; falling back to CPU solver.\n");
    }

    return solve(model, &cpu_params, results);
}