#include "reduction.h"

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Dense>

extern "C" void size(long **basis_ptr, const double eta, const long n, const long m)
{
    long i, j, q;
    const double eta_hat = (eta + 0.5) * 0.5;
    basis = MatrixXli::Zero(n, m);

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < m; ++j)
        {
            basis.coeffRef(i, j) = basis_ptr[i][j];
        }
    }

    computeGSO(basis);

    for (i = 0; i < n; ++i)
    {
        for (j = i - 1; j > -1; --j)
        {
            if (fabsl(mu.coeff(i, j)) > eta_hat)
            {
                q = round(mu.coeff(i, j));
                basis.row(i) -= q * basis.row(j);
                mu.row(i).head(j) -= static_cast<long double>(q) * mu.row(j).head(j);
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
