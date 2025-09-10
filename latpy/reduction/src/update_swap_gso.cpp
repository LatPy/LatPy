#include "reduction.h"

#include <iostream>

#include <eigen3/Eigen/Dense>

#include "core.h"

void updateSwapGSO(const long k, const long n)
{
    long double t;
    const long double nu = mu.coeff(k, k - 1);
    const long double BB = B.coeff(k) + nu * nu * B.coeff(k - 1);
    mu.coeffRef(k, k - 1) = (nu * B.coeff(k - 1)) / BB;
    B.coeffRef(k) = (B.coeffRef(k) * B.coeff(k - 1)) / BB;
    B.coeffRef(k - 1) = BB;

    mu.row(k - 1).head(k - 1).swap(mu.row(k).head(k - 1));
    for (long i = k + 1; i < n; ++i)
    {
        t = mu.coeff(i, k);
        mu.coeffRef(i, k) = mu.coeff(i, k - 1) - nu * t;
        mu.coeffRef(i, k - 1) = t + mu.coeff(k, k - 1) * mu.coeff(i, k);
    }
}
