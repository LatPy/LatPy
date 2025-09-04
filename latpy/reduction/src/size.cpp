#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

extern "C" void size(long **basis_ptr, const double eta, const long n, const long m)
{
    long i, j, q;
    basis = MatrixXli::Zero(n, m);

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }

    computeGSO(basis);

    for (i = 1; i < n; ++i)
    {
        for (j = i - 1; j > -1; --j)
        {
            if (fabsl(mu.coeff(i, j)) > eta)
            {
                q = round(mu.coeff(i, j));
                basis.row(i) -= q * basis.row(j);
                mu.row(i).head(j + 1) -= static_cast<long double>(q) * mu.row(j).head(j + 1);
            }
        }
    }

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis_ptr[i][j] = basis.coeff(i, j);
        }
    }
}
