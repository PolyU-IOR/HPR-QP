#ifndef HPRQP_VECTOR_H
#define HPRQP_VECTOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void vec_fill(double *x, size_t n, double value);
void vec_copy(double *dest, const double *src, size_t n);
void vec_axpby(double *y, const double *x, size_t n, double a, double b);
double vec_dot(const double *x, const double *y, size_t n);
double vec_norm2(const double *x, size_t n);
double vec_norm_inf(const double *x, size_t n);
void vec_project_box(double *y, const double *x, const double *lower, const double *upper, size_t n);

#ifdef __cplusplus
}
#endif

#endif
