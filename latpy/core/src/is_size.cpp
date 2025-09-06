#include "core.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

extern "C" bool isSize(long **basis_ptr, const long n, const long m)
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

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < i; ++j)
        {
            if (fabsl(mu.coeff(i, j)) > 0.5)
            {
                return false;
            }
        }
    }

    return true;
}
