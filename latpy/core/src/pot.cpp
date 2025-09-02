#include "core.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

extern "C" long double pot(long **basis_ptr, const long n, const long m)
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

    long double p = 1.0;
    for (i = 0; i < n; ++i)
    {
        p *= powl(B.coeff(i), static_cast<long double>(n - i));
    }
    return p;
}
