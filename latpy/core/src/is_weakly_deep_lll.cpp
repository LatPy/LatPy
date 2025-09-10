#include "core.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

extern "C" bool isWeaklyDeepLLL(long **basis_ptr, const double delta, const long n, const long m)
{
    long double C;

    long i, j, k;
    basis = MatrixXli::Zero(n, m);
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }
    computeGSO();

    for (k = 0; k < n; ++k)
    {
        C = basis.row(k).squaredNorm();
        for (i = 0; i < k; ++i)
        {
            if (delta * B.coeff(i) > C)
            {
                return false;
            }
            C -= mu.coeff(k, i) * mu.coeff(k, i) * B.coeff(i);
        }
    }

    return true;
}
