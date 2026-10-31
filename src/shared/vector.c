#include "shared/vector.h"

#include <math.h>

void vec_fill(double *x, size_t n, double value)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        x[i] = value;
    }
}

void vec_copy(double *dest, const double *src, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

void vec_axpby(double *y, const double *x, size_t n, double a, double b)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        y[i] = a * x[i] + b * y[i];
    }
}

double vec_dot(const double *x, const double *y, size_t n)
{
    size_t i;
    double sum = 0.0;
    for (i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

double vec_norm2(const double *x, size_t n)
{
    return sqrt(vec_dot(x, x, n));
}

double vec_norm_inf(const double *x, size_t n)
{
    size_t i;
    double max_abs = 0.0;
    for (i = 0; i < n; ++i) {
        double abs_x = fabs(x[i]);
        if (abs_x > max_abs) {
            max_abs = abs_x;
        }
    }
    return max_abs;
}

void vec_project_box(double *y, const double *x, const double *lower, const double *upper, size_t n)
{
    size_t i;
    for (i = 0; i < n; ++i) {
        double value = x[i];
        if (value < lower[i]) {
            value = lower[i];
        }
        if (value > upper[i]) {
            value = upper[i];
        }
        y[i] = value;
    }
}
