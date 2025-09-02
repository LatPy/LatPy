#include "core.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

extern "C" long double sl(long **basis_ptr, const long n, const long m)
{
    long i, j;

    basis = MatrixXli::Zero(n, m);
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }

    computeGSO(basis, mu, B);

    long double S = 0, T = 0;
    for (i = 0; i < n; ++i)
    {
        S += (i + 1) * logl(B.coeff(i));
        T += logl(B.coeff(i));
    }

    return 12 * (S - (n + 1) * T * 0.5) / (n * (n * n - 1));
}
