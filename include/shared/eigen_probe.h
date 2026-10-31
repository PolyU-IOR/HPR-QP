#ifndef HPRQP_EIGEN_PROBE_H
#define HPRQP_EIGEN_PROBE_H

#include <stddef.h>
#include <stdint.h>

/* Retry only: retain the original moment pair unless the first moment's
 * absolute stopping floor masks an unconverged/annihilated operator. Rebuild
 * both moments with this fixed nonsmooth sequence on that path only. */
#ifdef __CUDACC__
__host__ __device__
#endif
static inline double hprqp_eigen_retry_probe_value(size_t index)
{
    uint64_t bits = (uint64_t)index + UINT64_C(0x9e3779b97f4a7c15);
    bits = (bits ^ (bits >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    bits = (bits ^ (bits >> 27)) * UINT64_C(0x94d049bb133111eb);
    bits ^= bits >> 31;
    return 2.0 * (double)(bits >> 11) * 0x1.0p-53 - 1.0;
}

#endif
