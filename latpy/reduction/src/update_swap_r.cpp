#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

void updateSwapR(const long k, const long n)
{
    long double d;
    const long double a = R.coeff(k - 1, k - 1);
    const long double b = R.coeff(k, k);
    R.coeffRef(k - 1, k - 1) = sqrtl(R.coeff(k, k) * R.coeff(k, k) + R.coeff(k, k - 1) * R.coeff(k, k - 1));
    const long double t = a / R.coeff(k - 1, k - 1);
    R.coeffRef(k, k) *= t;
    const long double c = R.coeff(k, k - 1);
    R.coeffRef(k, k - 1) *= t;
    R.row(k).head(k - 1).swap(R.row(k - 1).head(k - 1));

    for (long i = k + 1; i < n; ++i)
    {
        d = R.coeff(i, k);
        R.coeffRef(i, k) = (R.coeff(i, k - 1) * b - R.coeff(i, k) * c) / R.coeff(k - 1, k - 1);
        R.coeffRef(i, k - 1) = (R.coeff(k - 1, k - 1) * d) / b + (R.coeff(i, k) * R.coeff(k, k - 1)) / R.coeff(k, k);
    }
}
