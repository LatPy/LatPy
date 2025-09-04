#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

long double sl(const long n)
{
    long double S = 0, T = 0;
    for (long i = 0; i < n; ++i)
    {
        S += (i + 1) * logl(B.coeff(i));
        T += logl(B.coeff(i));
    }

    return 12 * (S - (n + 1) * T * 0.5) / (n * (n * n - 1));
}
